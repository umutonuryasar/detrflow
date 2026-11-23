"""COCO category-id helpers shared by training and evaluation."""
from __future__ import annotations


def build_label_to_cat(cat_ids: list[int], num_labels: int) -> list[int]:
    """Map model label indices to COCO category ids by position.

    COCO category ids are not contiguous (1..90 with gaps). The checkpoint's
    label index i corresponds to the i-th COCO category in sorted id order,
    the same convention scripts/train.py uses. Mapping by position rather than
    by name matters: some checkpoints (e.g. PekingU/rtdetr_r50vd) use VOC-style
    names for 6 classes ("motorbike", "aeroplane", "sofa", "pottedplant",
    "diningtable", "tvmonitor"), and a name lookup silently sent those
    predictions to the wrong categories, costing about 5 AP on val2017.
    """
    cat_ids = sorted(cat_ids)
    if len(cat_ids) != num_labels:
        raise ValueError(
            f"Model has {num_labels} labels but the annotation file has "
            f"{len(cat_ids)} categories; cannot map labels to category ids."
        )
    return cat_ids
