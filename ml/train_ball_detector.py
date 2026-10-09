import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--model", type=Path, default=ROOT / "models/football-detector-pilot.pt")
    parser.add_argument("--name", default="football-ball-1280")
    args = parser.parse_args()
    data = ROOT / "data/processed/football-ball-only/data.yaml"
    if not data.exists():
        raise SystemExit("Run ml/prepare_ball_dataset.py first")
    model = YOLO(str(args.model))
    started = time.perf_counter()
    result = model.train(
        data=str(data), epochs=args.epochs, imgsz=args.image_size, batch=args.batch,
        device=0 if torch.cuda.is_available() else "cpu", workers=0,
        project=str(ROOT / "runs/training"), name=args.name, exist_ok=True,
        cache=False, plots=True, amp=True, seed=42, close_mosaic=2,
        degrees=2.0, translate=0.08, scale=0.35, fliplr=0.5, mosaic=0.8,
    )
    elapsed = time.perf_counter() - started
    best = Path(result.save_dir) / "weights/best.pt"
    destination = ROOT / "models/football-ball-detector.pt"
    if best.exists():
        shutil.copy2(best, destination)
    report = {
        "status": "passed" if best.exists() else "failed", "epochs": args.epochs,
        "image_size": args.image_size, "elapsed_seconds": round(elapsed, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "checkpoint": str(destination),
        "metrics": {key: float(value) for key, value in getattr(result, "results_dict", {}).items()},
    }
    report_path = ROOT / "runs/training" / f"{args.name}-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
