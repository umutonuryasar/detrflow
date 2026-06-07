"""Evaluate RT-DETR on COCO val2017 and report mAP.

Usage:
    python scripts/evaluate.py --config configs/rtdetr_r50_coco.yaml
    python scripts/evaluate.py --checkpoint checkpoints/epoch_012
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import torch
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from torchvision.datasets import CocoDetection
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}\nInstall pycocotools and torchvision.") from e


def build_val_loader(img_dir: str, ann_file: str, processor, batch_size: int, num_workers: int):
    base = CocoDetection(root=img_dir, annFile=ann_file)

    class _Wrapped(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            img, targets = base[idx]
            image_id = targets[0]["image_id"] if targets else base.ids[idx]
            encoding = processor(images=img, return_tensors="pt")
            return {
                "pixel_values": encoding["pixel_values"].squeeze(0),
                "image_id": image_id,
                "orig_size": torch.tensor([img.height, img.width]),
            }

        def __len__(self):
            return len(base)

    def collate(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "image_ids": [b["image_id"] for b in batch],
            "orig_sizes": torch.stack([b["orig_size"] for b in batch]),
        }

    return DataLoader(
        _Wrapped(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )


@torch.inference_mode()
def evaluate(cfg: dict, checkpoint: str | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg["training"].get("fp16", False) and device.type == "cuda"
    use_bf16 = cfg["training"].get("bf16", False) and device.type == "cuda"
    use_amp = use_fp16 or use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model_id = checkpoint or cfg["model"]["id"]
    processor = RTDetrImageProcessor.from_pretrained(model_id)
    model = RTDetrForObjectDetection.from_pretrained(
        model_id, torch_dtype=amp_dtype if use_amp else torch.float32
    ).to(device)
    model.eval()

    loader = build_val_loader(
        cfg["data"]["val_img"],
        cfg["data"]["val_ann"],
        processor,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    coco_gt = COCO(cfg["data"]["val_ann"])
    # Build HF label → COCO category_id mapping
    label2cat = {cat["name"]: cat["id"] for cat in coco_gt.cats.values()}

    results = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        orig_sizes = batch["orig_sizes"].to(device)

        with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            outputs = model(pixel_values=pixel_values)

        preds = processor.post_process_object_detection(
            outputs, target_sizes=orig_sizes, threshold=0.0
        )

        for image_id, pred in zip(batch["image_ids"], preds):
            for score, label, box in zip(pred["scores"], pred["labels"], pred["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": label2cat.get(model.config.id2label[label.item()], label.item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO [x,y,w,h]
                        "score": float(score),
                    }
                )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        tmp_path = f.name

    try:
        coco_dt = coco_gt.loadRes(tmp_path)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    finally:
        os.unlink(tmp_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rtdetr_r50_coco.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to saved checkpoint dir")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg, checkpoint=args.checkpoint)
