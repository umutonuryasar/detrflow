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

from inference.matcher import HungarianMatcher
from inference.criterion import SetCriterion

try:
    from pycocotools.coco import COCO
    from torchvision.datasets import CocoDetection
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}\nInstall pycocotools and torchvision.") from e

try:
    import wandb as _wandb_module
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_coco_dataset(img_dir: str, ann_file: str, processor: RTDetrImageProcessor):
    base = CocoDetection(root=img_dir, annFile=ann_file)

    cat_ids = sorted(base.coco.getCatIds())
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

    class _Wrapped(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            img, targets = base[idx]
            image_id = targets[0]["image_id"] if targets else 0
            annotations = {
                "image_id": image_id,
                "annotations": [
                    {
                        "bbox": t["bbox"],
                        "category_id": cat_id_to_idx[t["category_id"]],
                        "area": t["bbox"][2] * t["bbox"][3],
                        "iscrowd": t.get("iscrowd", 0),
                    }
                    for t in targets
                ],
            }
            encoding = processor(
                images=img,
                annotations=annotations,
                return_tensors="pt",
            )
            result = {}
            for k, v in encoding.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.squeeze(0)
                elif isinstance(v, list) and len(v) == 1:
                    result[k] = v[0]
                else:
                    result[k] = v
            return result

        def __len__(self):
            return len(base)

    return _Wrapped()


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    result: dict = {"pixel_values": pixel_values, "labels": labels}
    if "pixel_mask" in batch[0]:
        result["pixel_mask"] = torch.stack([b["pixel_mask"] for b in batch])
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict, resume: str | None = None) -> None:
    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # wandb init — wrapped so training works without credentials or installation
    _wandb = None
    if _WANDB_AVAILABLE:
        try:
            _wandb = _wandb_module.init(project="detrflow", config=cfg, resume="allow")
        except Exception as exc:
            print(f"[wandb] init failed ({exc}), continuing without logging.")

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

    num_classes: int = cfg["model"]["num_labels"]
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict: dict[str, float] = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
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

    # BF16 has sufficient dynamic range and does not risk underflow, so it does not need
    # loss scaling. GradScaler is only instantiated for FP16.
    scaler = GradScaler("cuda") if use_fp16 else None

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
    eval_every = cfg["training"].get("eval_every_n_epochs", 0)
    global_step = 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in lbl.items()} for lbl in batch["labels"]]

            with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pixel_mask = batch.get("pixel_mask")
                if pixel_mask is not None:
                    pixel_mask = pixel_mask.to(device)
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
                # Build the dict expected by SetCriterion from HF model outputs
                criterion_inputs = {
                    "logits": outputs.logits,
                    "pred_boxes": outputs.pred_boxes,
                }
                # targets need "class_labels" and "boxes" keys
                targets = [
                    {
                        "class_labels": lbl["class_labels"],
                        "boxes": lbl["boxes"],
                    }
                    for lbl in labels
                ]
                loss_dict = criterion(criterion_inputs, targets)
                total_loss = sum(weight_dict[k] * loss_dict[k] for k in loss_dict)
                loss = total_loss / grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if _wandb is not None:
                    _wandb.log({
                        "train/loss_ce":    loss_dict["loss_ce"].item(),
                        "train/loss_bbox":  loss_dict["loss_bbox"].item(),
                        "train/loss_giou":  loss_dict["loss_giou"].item(),
                        "train/loss_total": total_loss.item(),
                        "train/lr":         optimizer.param_groups[1]["lr"],
                        "train/step":       global_step,
                    })

            running_loss += loss.item() * grad_accum
            if step % 50 == 0:
                print(f"[epoch {epoch}/{epochs} step {step}] loss={running_loss/step:.4f}")

        # Flush any remaining accumulated gradients at end of epoch
        if step % grad_accum != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        epoch_log: dict = {"train/epoch_loss": running_loss / len(train_loader), "epoch": epoch}

        if eval_every > 0 and epoch % eval_every == 0:
            print(f"[epoch {epoch}] Running COCO val evaluation...")
            try:
                from evaluate import evaluate as _evaluate  # both scripts live in scripts/
                metrics = _evaluate(cfg)
            except Exception as exc:
                print(f"[eval] failed: {exc}")
                metrics = None
            if metrics is not None:
                epoch_log.update({f"eval/{k}": v for k, v in metrics.items()})
                print(f"[epoch {epoch}] AP={metrics['AP']:.4f}  AP50={metrics['AP50']:.4f}")

        if _wandb is not None:
            _wandb.log(epoch_log)

        if epoch % cfg["training"]["save_every_n_epochs"] == 0:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}")
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            print(f"Saved checkpoint → {ckpt_path}")


    if _wandb is not None:
        _wandb.finish()


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
