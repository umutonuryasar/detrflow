from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from inference.predictor import RTDetrPredictor
from api.schemas import Detection, BoundingBox, PredictResponse

MODEL_ID = os.getenv("MODEL_ID", "PekingU/rtdetr_r50vd")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

app = FastAPI(title="detrflow", description="RT-DETR object detection API", version="0.1.0")

_predictor: RTDetrPredictor | None = None


def get_predictor() -> RTDetrPredictor:
    global _predictor
    if _predictor is None:
        _predictor = RTDetrPredictor(
            model_id=MODEL_ID,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
    return _predictor


@app.on_event("startup")
async def _warm_up() -> None:
    get_predictor()


@app.post("/predict", response_model=PredictResponse, summary="Detect objects in an image")
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, …)"),
    threshold: float = Query(
        default=CONFIDENCE_THRESHOLD,
        ge=0.01,
        le=1.0,
        description="Override confidence threshold for this request",
    ),
) -> PredictResponse:
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Could not decode image")

    predictor = get_predictor()
    predictor.confidence_threshold = threshold
    raw_detections = predictor.predict(image)

    detections = [
        Detection(
            label=d["label"],
            score=d["score"],
            box=BoundingBox(**d["box"]),
        )
        for d in raw_detections
    ]

    return PredictResponse(
        detections=detections,
        model=MODEL_ID,
        image_width=image.width,
        image_height=image.height,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
