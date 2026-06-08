#!/usr/bin/env python3
"""
scripts/benchmark.py — detrflow inference benchmark
Measures FPS, latency (mean/p50/p95/p99), and peak GPU memory.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --model-id umutonuryasar/rtdetr-r50vd-coco-detrflow
    python scripts/benchmark.py --model-id umutonuryasar/rtdetr-r50vd-coco-detrflow --runs 200 --warmup 20
    python scripts/benchmark.py --image path/to/image.jpg
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference.predictor import RTDetrPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="detrflow inference benchmark")
    parser.add_argument(
        "--model-id",
        default=os.getenv("MODEL_ID", "umutonuryasar/rtdetr-r50vd-coco-detrflow"),
        help="HF model ID or local checkpoint path",
    )
    parser.add_argument("--image", default=None, help="Path to a test image (optional)")
    parser.add_argument("--runs", type=int, default=100, help="Number of timed inference runs")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs (excluded from stats)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 on CUDA")
    parser.add_argument("--size", type=int, nargs=2, default=[640, 640], metavar=("W", "H"),
                        help="Synthetic image size if no --image provided (default: 640 640)")
    return parser.parse_args()


def make_test_image(width: int, height: int) -> Image.Image:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def reset_gpu_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_gpu_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def run_benchmark(predictor: RTDetrPredictor, image: Image.Image, runs: int, warmup: int) -> dict:
    device = predictor.device

    # ── warmup ──────────────────────────────────────────────
    print(f"  Warming up ({warmup} runs)...", end=" ", flush=True)
    for _ in range(warmup):
        predictor.predict(image)
    if device == "cuda":
        torch.cuda.synchronize()
    print("done")

    # ── timed runs ──────────────────────────────────────────
    reset_gpu_stats()
    latencies_ms: list[float] = []

    print(f"  Benchmarking ({runs} runs)...", end=" ", flush=True)
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictor.predict(image)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    print("done")

    arr = np.array(latencies_ms)
    return {
        "mean_ms":   float(np.mean(arr)),
        "std_ms":    float(np.std(arr)),
        "min_ms":    float(np.min(arr)),
        "p50_ms":    float(np.percentile(arr, 50)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "p99_ms":    float(np.percentile(arr, 99)),
        "max_ms":    float(np.max(arr)),
        "fps":       1000.0 / float(np.mean(arr)),
        "peak_gpu_mb": peak_gpu_mb(),
    }


def print_results(stats: dict, args: argparse.Namespace, image: Image.Image,
                  predictor: RTDetrPredictor) -> None:
    w, h = image.size
    sep = "─" * 52

    print()
    print("╔" + "═" * 52 + "╗")
    print("║  detrflow — Inference Benchmark Results" + " " * 12 + "║")
    print("╠" + "═" * 52 + "╣")
    print(f"║  Model     : {args.model_id:<38}║")
    print(f"║  Device    : {predictor.device:<38}║")
    print(f"║  Precision : {'FP16' if predictor.dtype == torch.float16 else 'FP32':<38}║")
    print(f"║  Image     : {w}×{h} px{'':<31}║")
    print(f"║  Runs      : {args.runs} (+ {args.warmup} warmup){'':<24}║")
    print("╠" + "═" * 52 + "╣")
    print(f"║  Mean latency  : {stats['mean_ms']:>8.2f} ms{'':<23}║")
    print(f"║  Std           : {stats['std_ms']:>8.2f} ms{'':<23}║")
    print(f"║  Min           : {stats['min_ms']:>8.2f} ms{'':<23}║")
    print(f"║  p50           : {stats['p50_ms']:>8.2f} ms{'':<23}║")
    print(f"║  p95           : {stats['p95_ms']:>8.2f} ms{'':<23}║")
    print(f"║  p99           : {stats['p99_ms']:>8.2f} ms{'':<23}║")
    print(f"║  Max           : {stats['max_ms']:>8.2f} ms{'':<23}║")
    print("╠" + "═" * 52 + "╣")
    print(f"║  FPS           : {stats['fps']:>8.1f}{'':<31}║")
    if stats["peak_gpu_mb"] > 0:
        print(f"║  Peak GPU VRAM : {stats['peak_gpu_mb']:>8.1f} MB{'':<22}║")
    print("╚" + "═" * 52 + "╝")
    print()


def main() -> None:
    args = parse_args()

    print(f"\n[detrflow benchmark]")
    print(f"  Model    : {args.model_id}")
    print(f"  Device   : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Runs     : {args.runs} (warmup: {args.warmup})\n")

    # ── load image ──────────────────────────────────────────
    if args.image:
        print(f"  Loading image: {args.image}")
        image = Image.open(args.image).convert("RGB")
    else:
        w, h = args.size
        print(f"  Using synthetic {w}×{h} image")
        image = make_test_image(w, h)

    # ── load model ──────────────────────────────────────────
    print(f"  Loading model...", end=" ", flush=True)
    predictor = RTDetrPredictor(
        model_id=args.model_id,
        confidence_threshold=args.threshold,
        use_fp16=not args.no_fp16,
    )
    print("done\n")

    # ── run ─────────────────────────────────────────────────
    stats = run_benchmark(predictor, image, runs=args.runs, warmup=args.warmup)
    print_results(stats, args, image, predictor)


if __name__ == "__main__":
    main()