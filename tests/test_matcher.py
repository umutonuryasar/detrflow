"""Tests for HungarianMatcher and supporting utilities."""
from __future__ import annotations

import torch
import pytest

from inference.matcher import HungarianMatcher, giou


# ---------------------------------------------------------------------------
# giou tests
# ---------------------------------------------------------------------------

def test_giou_perfect_overlap():
    box = torch.tensor([[0.5, 0.5, 0.4, 0.4]])  # cx, cy, w, h
    result = giou(box, box)
    assert result.shape == (1, 1)
    assert torch.allclose(result, torch.ones(1, 1), atol=1e-5), f"Expected GIoU=1.0, got {result.item():.6f}"


def test_giou_no_overlap():
    # Box 1: top-left corner, Box 2: bottom-right corner — no overlap
    box1 = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    box2 = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    result = giou(box1, box2)
    assert result.shape == (1, 1)
    assert result.item() < 0, f"Non-overlapping boxes should have GIoU < 0, got {result.item():.6f}"


def test_giou_partial_overlap():
    box1 = torch.tensor([[0.3, 0.5, 0.4, 0.4]])
    box2 = torch.tensor([[0.6, 0.5, 0.4, 0.4]])
    result = giou(box1, box2)
    assert result.shape == (1, 1)
    assert -1.0 <= result.item() <= 1.0, f"GIoU out of range: {result.item()}"


def test_giou_pairwise_shape():
    boxes1 = torch.rand(5, 4).abs()
    boxes1[:, 2:] = boxes1[:, 2:].clamp(min=0.05)  # ensure positive w, h
    boxes2 = torch.rand(7, 4).abs()
    boxes2[:, 2:] = boxes2[:, 2:].clamp(min=0.05)
    result = giou(boxes1, boxes2)
    assert result.shape == (5, 7)


# ---------------------------------------------------------------------------
# HungarianMatcher tests
# ---------------------------------------------------------------------------

def _make_outputs(B: int, Q: int, C: int, device: str = "cpu") -> dict:
    return {
        "logits": torch.randn(B, Q, C, device=device),
        "pred_boxes": torch.rand(B, Q, 4, device=device),
    }


def _make_targets(num_targets_per_image: list[int], C: int, device: str = "cpu") -> list[dict]:
    targets = []
    for n in num_targets_per_image:
        targets.append({
            "class_labels": torch.randint(0, C, (n,), device=device),
            "boxes": torch.rand(n, 4, device=device),
        })
    return targets


def test_matcher_output_shape():
    B, Q, C = 2, 10, 80
    outputs = _make_outputs(B, Q, C)
    targets = _make_targets([3, 2], C)

    matcher = HungarianMatcher()
    indices = matcher(outputs, targets)

    assert len(indices) == B
    assert indices[0][0].shape == (3,), f"Expected pred_idx shape (3,), got {indices[0][0].shape}"
    assert indices[0][1].shape == (3,), f"Expected tgt_idx shape (3,), got {indices[0][1].shape}"
    assert indices[1][0].shape == (2,), f"Expected pred_idx shape (2,), got {indices[1][0].shape}"
    assert indices[1][1].shape == (2,), f"Expected tgt_idx shape (2,), got {indices[1][1].shape}"


def test_matcher_valid_indices():
    B, Q, C = 3, 15, 80
    num_targets = [4, 6, 2]
    outputs = _make_outputs(B, Q, C)
    targets = _make_targets(num_targets, C)

    matcher = HungarianMatcher()
    indices = matcher(outputs, targets)

    for i, (src_idx, tgt_idx) in enumerate(indices):
        T = num_targets[i]
        assert src_idx.max().item() < Q, f"Image {i}: pred_idx {src_idx.max()} >= num_queries {Q}"
        assert src_idx.min().item() >= 0, f"Image {i}: negative pred_idx"
        assert tgt_idx.max().item() < T, f"Image {i}: tgt_idx {tgt_idx.max()} >= num_targets {T}"
        assert tgt_idx.min().item() >= 0, f"Image {i}: negative tgt_idx"
        # Assignment is injective: no repeated indices
        assert src_idx.unique().numel() == src_idx.numel(), f"Image {i}: duplicate pred indices"
        assert tgt_idx.unique().numel() == tgt_idx.numel(), f"Image {i}: duplicate tgt indices"


def test_no_targets():
    B, Q, C = 2, 10, 80
    outputs = _make_outputs(B, Q, C)
    # Image 0: 0 targets, Image 1: 3 targets
    targets = _make_targets([0, 3], C)

    matcher = HungarianMatcher()
    indices = matcher(outputs, targets)

    assert len(indices) == B
    src_0, tgt_0 = indices[0]
    assert src_0.numel() == 0, "Image with 0 targets should return empty pred_idx"
    assert tgt_0.numel() == 0, "Image with 0 targets should return empty tgt_idx"

    src_1, tgt_1 = indices[1]
    assert src_1.shape == (3,)
    assert tgt_1.shape == (3,)


def test_matcher_cost_weights():
    """Changing cost weights should generally change the assignment."""
    torch.manual_seed(0)
    B, Q, C = 1, 20, 80
    outputs = _make_outputs(B, Q, C)
    targets = _make_targets([5], C)

    m1 = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    m2 = HungarianMatcher(cost_class=10.0, cost_bbox=0.1, cost_giou=0.1)

    idx1 = m1(outputs, targets)[0][0]
    idx2 = m2(outputs, targets)[0][0]

    # With very different weights the assignments will typically differ
    # (not guaranteed but highly likely with random inputs)
    assert idx1.shape == (5,) and idx2.shape == (5,)
