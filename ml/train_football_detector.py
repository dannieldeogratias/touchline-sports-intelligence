import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the local football detector.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/external/football-image-dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--name", default="football-pilot")
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo26n.pt")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset configuration not found: {args.data}")
    dataset_root = args.data.resolve().parent
    runtime_yaml = ROOT / ".cache" / "football-image-dataset.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    runtime_yaml.write_text(
        f"path: {dataset_root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n  1: player\n  2: referee\n",
        encoding="utf-8",
    )
    model = YOLO(str(args.model))
    started = time.perf_counter()
    result = model.train(
        data=str(runtime_yaml), epochs=args.epochs, imgsz=args.image_size,
        batch=args.batch, device=0 if torch.cuda.is_available() else "cpu",
        workers=0, project=str(ROOT / "runs/training"), name=args.name,
        exist_ok=True, cache=False, plots=True, amp=True, seed=42,
    )
    elapsed = time.perf_counter() - started
    best = Path(result.save_dir) / "weights" / "best.pt"
    promoted_copy = ROOT / "models" / "football-detector-pilot.pt"
    if best.exists():
        shutil.copy2(best, promoted_copy)
    metrics = getattr(result, "results_dict", {})
    report = {
        "status": "passed" if best.exists() else "failed",
        "dataset": str(args.data.resolve()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch": args.batch,
        "elapsed_seconds": round(elapsed, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else 0,
        "best_checkpoint": best.as_posix(),
        "pilot_checkpoint": promoted_copy.as_posix(),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "limitations": [
            "Dataset filenames do not preserve match identity, so match-grouped isolation is unverified.",
            "The provided validation set mixes broadcast footage and PES 21 gameplay.",
            "This checkpoint is experimental and is not automatically promoted to the product worker.",
        ],
    }
    report_path = ROOT / "runs/training" / f"{args.name}-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
