from __future__ import annotations

import os

import gradio as gr
from PIL import Image

from inference.predictor import RTDetrPredictor
from inference.visualizer import draw_detections

MODEL_ID = os.getenv("MODEL_ID", "umutonuryasar/rtdetr-r50vd-coco-detrflow")

predictor = RTDetrPredictor(model_id=MODEL_ID)


def detect(image: Image.Image, threshold: float) -> tuple[Image.Image, str]:
    if image is None:
        return None, "No image provided."

    predictor.confidence_threshold = threshold
    detections = predictor.predict(image)
    annotated = draw_detections(image, detections)

    if not detections:
        summary = "No objects detected above threshold."
    else:
        lines = [f"**{d['label']}** — {d['score']:.2%}" for d in detections]
        summary = f"Found **{len(detections)}** object(s):\n\n" + "\n".join(lines)

    return annotated, summary


with gr.Blocks(title="detrflow — RT-DETR Object Detection") as demo:
    gr.Markdown("# detrflow — RT-DETR Object Detection\nUpload an image to detect objects using `PekingU/rtdetr_r50vd`.")

    with gr.Row():
        with gr.Column():
            inp_image = gr.Image(type="pil", label="Input Image")
            threshold = gr.Slider(0.01, 1.0, value=0.5, step=0.01, label="Confidence threshold")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            out_image = gr.Image(type="pil", label="Annotated Image")
            out_text = gr.Markdown(label="Detections")

    btn.click(fn=detect, inputs=[inp_image, threshold], outputs=[out_image, out_text])
    inp_image.change(fn=detect, inputs=[inp_image, threshold], outputs=[out_image, out_text])

if __name__ == "__main__":
    demo.launch()