from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left edge (pixels)")
    y1: float = Field(..., description="Top edge (pixels)")
    x2: float = Field(..., description="Right edge (pixels)")
    y2: float = Field(..., description="Bottom edge (pixels)")


class Detection(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox


class PredictResponse(BaseModel):
    detections: list[Detection]
    model: str
    image_width: int
    image_height: int
