"""Tests for label-index -> COCO category-id mapping."""
from __future__ import annotations

import pytest

from inference.categories import build_label_to_cat

# Real COCO 2017 category ids: 80 classes, non-contiguous in 1..90.
COCO_CAT_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def test_maps_by_sorted_position():
    mapping = build_label_to_cat(list(reversed(COCO_CAT_IDS)), num_labels=80)
    assert mapping == COCO_CAT_IDS
    assert mapping[0] == 1 and mapping[79] == 90


def test_vocstyle_name_classes_get_correct_ids():
    # Indices whose checkpoint names differ from COCO names in PekingU/rtdetr_r50vd.
    # A name lookup fell back to the raw index for these, e.g. motorbike -> 3 (car).
    mapping = build_label_to_cat(COCO_CAT_IDS, num_labels=80)
    expected = {3: 4, 4: 5, 57: 63, 58: 64, 60: 67, 62: 72}  # motorcycle, airplane, couch, potted plant, dining table, tv
    for idx, cat_id in expected.items():
        assert mapping[idx] == cat_id
        assert mapping[idx] != idx


def test_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="80 labels"):
        build_label_to_cat(COCO_CAT_IDS[:79], num_labels=80)
