from __future__ import annotations

import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

DEFAULT_MODEL = "PekingU/rtdetr_r50vd"


class RTDetrPredictor:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        use_fp16: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.dtype = torch.float16 if (use_fp16 and self.device == "cuda") else torch.float32

        self.processor = RTDetrImageProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> list[dict]:
        """Run inference on a PIL image.

        Returns a list of dicts with keys: label, score, box (xyxy floats, pixel coords).
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
                  for k, v in inputs.items()}

        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=self.device)  # (H, W)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold,
        )[0]

        detections = []
        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            detections.append(
                {
                    "label": self.model.config.id2label[label_id.item()],
                    "score": round(score.item(), 4),
                    "box": {
                        "x1": round(box[0].item(), 2),
                        "y1": round(box[1].item(), 2),
                        "x2": round(box[2].item(), 2),
                        "y2": round(box[3].item(), 2),
                    },
                }
            )

        return sorted(detections, key=lambda d: d["score"], reverse=True)
