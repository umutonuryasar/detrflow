"""Fine-tune RT-DETR on COCO.

Usage:
    python scripts/train.py --config configs/rtdetr_r50_coco.yaml
"""
from __future__ import annotations

import argparse
import math
import os
import random

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

try:
    from pycocotools.coco import COCO
    from torchvision.datasets import CocoDetection
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}\nInstall pycocotools and torchvision.") from e


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_coco_dataset(img_dir: str, ann_file: str, processor: RTDetrImageProcessor):
    base = CocoDetection(root=img_dir, annFile=ann_file)

    class _Wrapped(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            img, targets = base[idx]
            annotations = [
                {
                    "bbox": t["bbox"],           # [x, y, w, h]
                    "category_id": t["category_id"],
                    "image_id": t["image_id"],
                }
                for t in targets
            ]
            encoding = processor(
                images=img,
                annotations=annotations,
                return_tensors="pt",
            )
            return {k: v.squeeze(0) for k, v in encoding.items()}

        def __len__(self):
            return len(base)

    return _Wrapped()


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = cfg["training"]["fp16"] and device.type == "cuda"

    processor = RTDetrImageProcessor.from_pretrained(cfg["model"]["id"])
    model = RTDetrForObjectDetection.from_pretrained(
        cfg["model"]["id"],
        num_labels=cfg["model"]["num_labels"],
        ignore_mismatched_sizes=True,
    ).to(device)

    if cfg["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # Split params: backbone gets a lower LR
    factor = cfg["training"]["optimizer"]["backbone_lr_factor"]
    base_lr = cfg["training"]["optimizer"]["lr"]
    backbone_params, rest_params = [], []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            rest_params.append(param)

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": base_lr * factor},
            {"params": rest_params,     "lr": base_lr},
        ],
        weight_decay=cfg["training"]["optimizer"]["weight_decay"],
    )

    epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"]["scheduler"]["warmup_epochs"]
    min_lr = cfg["training"]["scheduler"]["min_lr"]

    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    scaler = GradScaler(enabled=use_fp16)

    train_ds = build_coco_dataset(
        cfg["data"]["train_img"], cfg["data"]["train_ann"], processor
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    grad_accum = cfg["training"]["grad_accum_steps"]
    clip_norm = cfg["training"]["clip_grad_norm"]

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in batch["labels"]]

            with autocast(enabled=use_fp16):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum

            scaler.scale(loss).backward()

            if step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_accum
            if step % 50 == 0:
                print(f"[epoch {epoch}/{epochs} step {step}] loss={running_loss/step:.4f}")

        scheduler.step()

        if epoch % cfg["training"]["save_every_n_epochs"] == 0:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}")
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            print(f"Saved checkpoint → {ckpt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rtdetr_r50_coco.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
