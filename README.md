# detrflow

End-to-end toolkit for **RT-DETR** object detection — from fine-tuning on COCO to serving predictions via a REST API and an interactive Gradio demo.

## Overview

| Component              | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `inference/`           | `RTDetrPredictor` — wraps HuggingFace RT-DETR with FP16 support  |
| `api/`                 | FastAPI service with `/predict` (multipart upload) and `/health` |
| `demo/`                | Gradio web app hosted on HuggingFace Spaces                      |
| `scripts/train.py`     | COCO fine-tuning with AMP, gradient accumulation, and cosine LR  |
| `scripts/evaluate.py`  | COCO val2017 mAP evaluation via pycocotools                      |
| `scripts/benchmark.py` | Latency/FPS benchmark with p50/p95/p99 percentile reporting      |

## Requirements

- Python 3.11+
- PyTorch 2.2+ (CUDA recommended for training)

```bash
pip install -r requirements.txt
```

## Quick Start

### Interactive Demo

```bash
python demo/app.py
```

Opens a Gradio interface at `http://localhost:7860`. Upload an image and adjust the confidence threshold slider to filter detections.

The live demo is also available on [HuggingFace Spaces](https://huggingface.co/spaces/umutonuryasar/detrflow).

### REST API (Docker)

```bash
docker compose up --build
```

The API starts on port `8000`. Test it:

```bash
# Health check
curl http://localhost:8000/health

# Run detection on an image
curl -X POST http://localhost:8000/predict \
     -F "file=@image.jpg" \
     -F "threshold=0.5" | python3 -m json.tool
```

**Response schema:**

```json
{
  "detections": [
    {
      "label": "dog",
      "score": 0.9341,
      "box": { "x1": 42.1, "y1": 100.5, "x2": 310.7, "y2": 480.2 }
    }
  ],
  "model": "PekingU/rtdetr_r50vd",
  "image_width": 640,
  "image_height": 480
}
```

**Environment variables:**

| Variable               | Default                | Description                        |
| ---------------------- | ---------------------- | ---------------------------------- |
| `MODEL_ID`             | `PekingU/rtdetr_r50vd` | HuggingFace model ID or local path |
| `CONFIDENCE_THRESHOLD` | `0.5`                  | Default detection threshold        |
| `HF_TOKEN`             | _(empty)_              | Required for private HF models     |

### Python API

```python
from PIL import Image
from inference.predictor import RTDetrPredictor
from inference.visualizer import draw_detections

predictor = RTDetrPredictor(
    model_id="umutonuryasar/rtdetr-r50vd-coco-detrflow",
    confidence_threshold=0.5,
    use_fp16=True,   # automatically disabled on CPU
)

image = Image.open("image.jpg")
detections = predictor.predict(image)
annotated = draw_detections(image, detections)
annotated.save("output.jpg")

for d in detections:
    print(f"{d['label']:15s}  {d['score']:.2%}  {d['box']}")
```

## Fine-Tuning on COCO

### 1. Prepare data

Download COCO 2017 and place it under `data/coco/`:

```
data/coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

### 2. Configure

Edit `configs/rtdetr_r50_coco.yaml`. Key settings:

```yaml
model:
  id: PekingU/rtdetr_r50vd   # base checkpoint
  num_labels: 80             # COCO classes

training:
  epochs: 12
  batch_size: 16             # A100 40 GB with bf16
  grad_accum_steps: 4        # effective batch size = 64
  bf16: true                 # native bfloat16 on Ampere GPUs
  gradient_checkpointing: false
```

### 3. Train

```bash
python scripts/train.py --config configs/rtdetr_r50_coco.yaml
```

Resume from a checkpoint:

```bash
python scripts/train.py --config configs/rtdetr_r50_coco.yaml \
                        --resume checkpoints/epoch_006
```

Checkpoints are saved to `checkpoints/epoch_NNN/` at the interval set by `save_every_n_epochs`.

### 4. Evaluate

```bash
# Evaluate the base model
python scripts/evaluate.py --config configs/rtdetr_r50_coco.yaml

# Evaluate a fine-tuned checkpoint
python scripts/evaluate.py --checkpoint checkpoints/epoch_012
```

Reports standard COCO mAP metrics (AP@[.50:.95], AP50, AP75, APs, APm, APl).

### 5. Benchmark

```bash
python scripts/benchmark.py --model-id umutonuryasar/rtdetr-r50vd-coco-detrflow

# Custom image and run count
python scripts/benchmark.py --image photo.jpg --runs 200 --warmup 20

# Disable FP16
python scripts/benchmark.py --no-fp16
```

Sample output:

```
╔════════════════════════════════════════════════════╗
║  detrflow — Inference Benchmark Results            ║
╠════════════════════════════════════════════════════╣
║  Model     : umutonuryasar/rtdetr-r50vd-coco-...  ║
║  Device    : cuda                                  ║
║  Precision : FP16                                  ║
║  Image     : 640×640 px                            ║
║  Runs      : 100 (+ 10 warmup)                     ║
╠════════════════════════════════════════════════════╣
║  Mean latency  :    18.43 ms                       ║
║  p50           :    18.21 ms                       ║
║  p95           :    19.87 ms                       ║
║  p99           :    21.04 ms                       ║
╠════════════════════════════════════════════════════╣
║  FPS           :      54.3                         ║
║  Peak GPU VRAM :   1842.0 MB                       ║
╚════════════════════════════════════════════════════╝
```

## Project Structure

```
detrflow/
├── api/
│   ├── Dockerfile
│   ├── main.py          # FastAPI application
│   └── schemas.py       # Pydantic response models
├── configs/
│   └── rtdetr_r50_coco.yaml
├── demo/
│   └── app.py           # Gradio interface
├── inference/
│   ├── predictor.py     # RTDetrPredictor
│   └── visualizer.py    # Bounding-box rendering
├── notebooks/
│   └── detrflow_training.ipynb
├── scripts/
│   ├── benchmark.py
│   ├── evaluate.py
│   └── train.py
├── docker-compose.yml
└── requirements.txt
```

## License

[MIT](LICENSE)
