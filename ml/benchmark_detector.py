import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the local YOLO detector on a synthetic frame.")
    parser.add_argument("--model", default="models/yolo26n.pt")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path if model_path.exists() else model_path.name))
    if not model_path.exists():
        Path(model.ckpt_path).replace(model_path)
        model = YOLO(str(model_path))

    device = 0 if torch.cuda.is_available() else "cpu"
    frame = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
    frame[80:600, 20:620] = (38, 112, 52)
    model.predict(frame, imgsz=args.image_size, device=device, verbose=False)
    timings = []
    for _ in range(args.runs):
        started = time.perf_counter()
        model.predict(frame, imgsz=args.image_size, device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)

    result = {
        "model": model_path.as_posix(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "image_size": args.image_size,
        "runs": args.runs,
        "median_ms": round(statistics.median(timings), 2),
        "mean_ms": round(statistics.mean(timings), 2),
        "fps_from_median": round(1000 / statistics.median(timings), 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else 0,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
