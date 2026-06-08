"""Set-based loss criterion for DETR-family object detection models.

Computes classification, L1, and GIoU losses over the optimal bipartite
matching produced by HungarianMatcher, as described in:
  - DETR: End-to-End Object Detection with Transformers, Carion et al. 2020
  - RT-DETR: DETRs Beat YOLOs on Real-time Object Detection, Zhao et al. 2022
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import HungarianMatcher, giou


class SetCriterion(nn.Module):
    """Loss criterion that pairs predictions with targets via bipartite matching.

    Computes three loss terms:
      - loss_ce:    Cross-entropy over all queries (no-object class down-weighted).
      - loss_bbox:  L1 loss on normalized cxcywh boxes for matched pairs.
      - loss_giou:  1 − GIoU for matched pairs.

    Reference:
        DETR: End-to-End Object Detection with Transformers, Carion et al. 2020
        RT-DETR: DETRs Beat YOLOs on Real-time Object Detection, Zhao et al. 2022

    Args:
        num_classes: Number of foreground object classes (no-object is index num_classes).
        matcher: HungarianMatcher instance.
        weight_dict: Per-loss scalar weights, e.g. {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}.
        eos_coef: Down-weighting factor for the no-object (background) class.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float],
        eos_coef: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        # Class weights: 1 for foreground, eos_coef for the no-object slot
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[num_classes] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(
        self, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (batch_idx, query_idx) for all matched predictions."""
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def loss_labels(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        logits: torch.Tensor = outputs["logits"]  # (B, Q, C)
        B, Q, _ = logits.shape
        device = logits.device

        # Default: every query is background (num_classes)
        target_classes = torch.full((B, Q), self.num_classes, dtype=torch.long, device=device)

        for i, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_classes[i, src_idx] = targets[i]["class_labels"][tgt_idx]

        # logits: (B, Q, C+1) → (B, C+1, Q) for cross_entropy
        return F.cross_entropy(
            logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        )

    def loss_boxes(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        pred_boxes: torch.Tensor = outputs["pred_boxes"]  # (B, Q, 4)
        batch_idx, src_idx = self._get_src_permutation_idx(indices)

        src_boxes = pred_boxes[batch_idx, src_idx]  # (total_matched, 4)
        tgt_boxes = torch.cat([
            t["boxes"][j] for t, (_, j) in zip(targets, indices)
        ])  # (total_matched, 4)

        num_matched = src_boxes.shape[0]
        if num_matched == 0:
            return pred_boxes.sum() * 0.0

        return F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_matched

    def loss_giou(
        self,
        outputs: dict,
        targets: list[dict],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        pred_boxes: torch.Tensor = outputs["pred_boxes"]  # (B, Q, 4)
        batch_idx, src_idx = self._get_src_permutation_idx(indices)

        src_boxes = pred_boxes[batch_idx, src_idx]  # (total_matched, 4)
        tgt_boxes = torch.cat([
            t["boxes"][j] for t, (_, j) in zip(targets, indices)
        ])  # (total_matched, 4)

        num_matched = src_boxes.shape[0]
        if num_matched == 0:
            return pred_boxes.sum() * 0.0

        # giou() returns pairwise (N, M); we need diagonal for matched pairs
        giou_diag = torch.diagonal(giou(src_boxes, tgt_boxes))
        return (1.0 - giou_diag).sum() / num_matched

    def forward(
        self,
        outputs: dict,
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        """Compute the set-based losses.

        Args:
            outputs: dict with "logits" (B, Q, C) and "pred_boxes" (B, Q, 4).
            targets: list of B dicts with "class_labels" and "boxes".

        Returns:
            dict with keys "loss_ce", "loss_bbox", "loss_giou".
        """
        indices = self.matcher(outputs, targets)

        return {
            "loss_ce":    self.loss_labels(outputs, targets, indices),
            "loss_bbox":  self.loss_boxes(outputs, targets, indices),
            "loss_giou":  self.loss_giou(outputs, targets, indices),
        }
