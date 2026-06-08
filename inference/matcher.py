"""Hungarian Matcher for DETR-family object detection models.

Implements bipartite matching between predicted and ground-truth boxes using
the Hungarian algorithm, as described in:
  - DETR: End-to-End Object Detection with Transformers, Carion et al. 2020
  - RT-DETR: DETRs Beat YOLOs on Real-time Object Detection, Zhao et al. 2022
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Generalized IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) cxcywh normalized
        boxes2: (M, 4) cxcywh normalized

    Returns:
        (N, M) GIoU matrix with values in [-1, 1]
    """
    b1 = box_cxcywh_to_xyxy(boxes1)  # (N, 4)
    b2 = box_cxcywh_to_xyxy(boxes2)  # (M, 4)

    # Intersection
    inter_x1 = torch.max(b1[:, None, 0], b2[None, :, 0])  # (N, M)
    inter_y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = torch.min(b1[:, None, 3], b2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h  # (N, M)

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])  # (N,)
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])  # (M,)
    union_area = area1[:, None] + area2[None, :] - inter_area  # (N, M)

    iou = inter_area / union_area.clamp(min=1e-6)

    # Enclosing box
    enc_x1 = torch.min(b1[:, None, 0], b2[None, :, 0])
    enc_y1 = torch.min(b1[:, None, 1], b2[None, :, 1])
    enc_x2 = torch.max(b1[:, None, 2], b2[None, :, 2])
    enc_y2 = torch.max(b1[:, None, 3], b2[None, :, 3])

    enc_area = ((enc_x2 - enc_x1) * (enc_y2 - enc_y1)).clamp(min=1e-6)  # (N, M)

    return iou - (enc_area - union_area) / enc_area


class HungarianMatcher(nn.Module):
    """Optimal bipartite matching between predictions and ground-truth targets.

    Solves the assignment problem by constructing a cost matrix from three
    terms — classification probability, L1 box distance, and GIoU — and
    running the Hungarian algorithm via scipy.

    Reference:
        DETR: End-to-End Object Detection with Transformers, Carion et al. 2020
        RT-DETR: DETRs Beat YOLOs on Real-time Object Detection, Zhao et al. 2022

    Args:
        cost_class: Weight for the classification cost term.
        cost_bbox: Weight for the L1 bounding-box cost term.
        cost_giou: Weight for the GIoU cost term.
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: list[dict],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run Hungarian matching for a batch.

        Args:
            outputs: dict with
                "logits"     — (B, num_queries, num_classes)
                "pred_boxes" — (B, num_queries, 4) cxcywh normalized
            targets: list of B dicts, each with
                "class_labels" — LongTensor (num_targets,)
                "boxes"        — FloatTensor (num_targets, 4) cxcywh normalized

        Returns:
            List of (pred_idx, tgt_idx) LongTensor pairs, one per image.
        """
        logits: torch.Tensor = outputs["logits"]      # (B, Q, C)
        pred_boxes: torch.Tensor = outputs["pred_boxes"]  # (B, Q, 4)

        indices: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, target in enumerate(targets):
            tgt_labels: torch.Tensor = target["class_labels"]  # (T,)
            tgt_boxes: torch.Tensor = target["boxes"]           # (T, 4)
            num_targets = tgt_labels.shape[0]

            if num_targets == 0:
                device = logits.device
                empty = torch.zeros(0, dtype=torch.long, device=device)
                indices.append((empty, empty))
                continue

            # (Q, C) → softmax probabilities, then index by target class
            probs = logits[i].softmax(-1)                 # (Q, C)
            cost_class = -probs[:, tgt_labels]            # (Q, T)

            # L1 distance between every query box and every target box
            pred_i = pred_boxes[i]                        # (Q, 4)
            cost_bbox = torch.cdist(pred_i, tgt_boxes, p=1)  # (Q, T)

            # Negative GIoU
            cost_giou_mat = -giou(pred_i, tgt_boxes)     # (Q, T)

            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou_mat
            )  # (Q, T)

            row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())

            device = logits.device
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long, device=device),
                torch.as_tensor(col_ind, dtype=torch.long, device=device),
            ))

        return indices
