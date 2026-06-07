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
from torch.amp import GradScaler, autocast
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
            image_id = targets[0]["image_id"] if targets else 0
            annotations = {
                "image_id": image_id,
                "annotations": [
                    {"bbox": t["bbox"], "category_id": t["category_id"]}
                    for t in targets
                ],
            }
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

def train(cfg: dict, resume: str | None = None) -> None:
    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = cfg["training"].get("fp16", False) and device.type == "cuda"
    use_bf16: bool = cfg["training"].get("bf16", False) and device.type == "cuda"
    use_amp: bool = use_fp16 or use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model_src = resume if resume else cfg["model"]["id"]
    processor = RTDetrImageProcessor.from_pretrained(model_src)
    model = RTDetrForObjectDetection.from_pretrained(
        model_src,
        num_labels=cfg["model"]["num_labels"],
        ignore_mismatched_sizes=(resume is None),
    ).to(device)

    # Infer which epoch to start from when resuming
    start_epoch = 1
    if resume:
        import re as _re
        m = _re.search(r"epoch_(\d+)", os.path.basename(resume.rstrip("/\\")))
        if m:
            start_epoch = int(m.group(1)) + 1
        print(f"Resuming from {resume}, starting at epoch {start_epoch}")

    if cfg["training"]["gradient_checkpointing"]:
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled.")
        except ValueError:
            print("Gradient checkpointing not supported by this model, skipping.")

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

    scaler = GradScaler("cuda", enabled=use_fp16)  # GradScaler only needed for fp16, not bf16

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

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in batch["labels"]]

            with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
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

        # Flush any remaining accumulated gradients at end of epoch
        if step % grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

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
    parser.add_argument("--resume", default=None, help="Path to checkpoint dir to resume training from")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume=args.resume)
