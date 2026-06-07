from __future__ import annotations

import random
from PIL import Image, ImageDraw, ImageFont


def _get_color(label: str) -> tuple[int, int, int]:
    rng = random.Random(hash(label) & 0xFFFFFFFF)
    h = rng.random()
    # HSV → RGB with S=0.7, V=0.9
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, 0.70, 0.90)
    return int(r * 255), int(g * 255), int(b * 255)


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: list[dict],
    line_width: int = 2,
    font_size: int = 14,
) -> Image.Image:
    """Overlay bounding boxes, labels, and confidence scores on *image*.

    *detections* is the list returned by RTDetrPredictor.predict().
    Returns a new RGB image (original is not mutated).
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    font = _load_font(font_size)

    for det in detections:
        label: str = det["label"]
        score: float = det["score"]
        box = det["box"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        color = _get_color(label)
        fill_rgba = (*color, 40)  # translucent fill

        draw.rectangle([x1, y1, x2, y2], outline=color, fill=fill_rgba, width=line_width)

        text = f"{label} {score:.2f}"
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Background pill behind text
        pad = 3
        draw.rectangle(
            [x1, y1 - text_h - pad * 2, x1 + text_w + pad * 2, y1],
            fill=(*color, 220),
        )
        draw.text((x1 + pad, y1 - text_h - pad), text, fill=(255, 255, 255), font=font)

    return out
