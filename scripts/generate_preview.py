"""
generate_preview.py
-------------------
Usage (run from the detrflow repo root):
    python scripts/generate_preview.py

This script generates a preview image showcasing RT-DETR's performance on four visually diverse COCO val2017 images.
It fetches the images from public URLs, runs inference using the RT-DETR model, and creates a 2×2 grid of annotated images with captions indicating the scene type and number of detections.
The final output is saved as `detrflow-preview.png`, which can be used for portfolio or promotional purposes.
Make sure to have the required dependencies installed (Pillow, and the RT-DETR model files) before running the script.
"""

from __future__ import annotations

import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Use detrflow's own modules ──────────────────────────────────────────────
from inference.predictor import RTDetrPredictor
from inference.visualizer import draw_detections

# ── Four visually diverse COCO val2017 images (public URLs) ─────────────────
SAMPLE_IMAGES = [
    # Crowded street scene — many people + vehicles
    ("https://farm5.staticflickr.com/4032/4322948498_e994f8f0a5_z.jpg",  "Street scene"),
    # Kitchen / indoor objects
    ("https://farm4.staticflickr.com/3153/2970773875_86e5b79042_z.jpg",  "Kitchen"),
    # Animals — horses
    ("https://farm4.staticflickr.com/3488/3773948627_0e9f359ff6_z.jpg",  "Animals"),
    # Sports — people in action
    ("https://farm9.staticflickr.com/8035/8097037927_0e7ccca37b_z.jpg",  "Sports"),
]

GRID_COLS   = 2
GRID_ROWS   = 2
CELL_W      = 640
CELL_H      = 480
PADDING     = 8
LABEL_H     = 28          # caption bar height at bottom of each cell
BG_COLOR    = (15, 17, 23)
CAPTION_BG  = (30, 33, 42)
ACCENT      = (0, 212, 170)   # teal — matches typical ML portfolio aesthetics


def fetch_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return Image.open(BytesIO(resp.read())).convert("RGB")


def resize_crop(img: Image.Image, w: int, h: int) -> Image.Image:
    """Center-crop to w×h after scaling to fill."""
    ratio = max(w / img.width, h / img.height)
    new_w, new_h = int(img.width * ratio), int(img.height * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top  = (new_h - h) // 2
    return img.crop((left, top, left + w, top + h))


def add_caption(cell: Image.Image, text: str, n_dets: int) -> Image.Image:
    """Paste a caption bar at the bottom of the cell."""
    draw = ImageDraw.Draw(cell)
    bar_y = CELL_H - LABEL_H
    draw.rectangle([0, bar_y, CELL_W, CELL_H], fill=(*CAPTION_BG, 220))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    caption = f"{text}  •  {n_dets} detections"
    draw.text((10, bar_y + 7), caption, fill=ACCENT, font=font)
    return cell


def make_header(total_w: int) -> Image.Image:
    """Top banner with model name and AP score."""
    h = 52
    banner = Image.new("RGB", (total_w, h), BG_COLOR)
    draw   = ImageDraw.Draw(banner)
    try:
        font_big  = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font_big = font_small = ImageFont.load_default()

    draw.text((16, 10), "detrflow", fill=(255, 255, 255), font=font_big)
    draw.text((110, 14), "RT-DETR  |  COCO val2017  AP = 47.9", fill=ACCENT, font=font_small)
    # thin accent line at bottom of banner
    draw.line([(0, h - 2), (total_w, h - 2)], fill=ACCENT, width=2)
    return banner


def main() -> None:
    predictor = RTDetrPredictor(confidence_threshold=0.45)
    print(f"Model loaded on {predictor.device}")

    cells: list[Image.Image] = []

    for url, caption in SAMPLE_IMAGES:
        print(f"  Fetching: {caption} …", end=" ", flush=True)
        try:
            img = fetch_image(url)
        except Exception as exc:
            print(f"SKIP ({exc})")
            # fallback: blank cell
            img = Image.new("RGB", (CELL_W, CELL_H), (40, 40, 40))
            cells.append(img)
            continue

        img = resize_crop(img, CELL_W, CELL_H)
        detections = predictor.predict(img)
        print(f"{len(detections)} detections")

        annotated = draw_detections(img, detections)
        annotated = add_caption(annotated, caption, len(detections))
        cells.append(annotated)

    # ── Assemble 2×2 grid ───────────────────────────────────────────────────
    total_w = GRID_COLS * CELL_W + (GRID_COLS + 1) * PADDING
    total_h = GRID_ROWS * CELL_H + (GRID_ROWS + 1) * PADDING

    header  = make_header(total_w)
    canvas  = Image.new("RGB", (total_w, total_h + header.height), BG_COLOR)
    canvas.paste(header, (0, 0))

    for idx, cell in enumerate(cells):
        row = idx // GRID_COLS
        col = idx  % GRID_COLS
        x = PADDING + col * (CELL_W + PADDING)
        y = header.height + PADDING + row * (CELL_H + PADDING)
        canvas.paste(cell, (x, y))

    out_path = Path("detrflow-preview.png")
    canvas.save(out_path, format="PNG", optimize=True)
    print(f"\n✓  Saved → {out_path.resolve()}")
    print("  Copy to your Academic Pages repo: /images/detrflow-preview.png")


if __name__ == "__main__":
    main()