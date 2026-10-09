import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "synthetic-yolo-smoke"
RUNS = ROOT / "runs" / "training"


def create_dataset() -> Path:
    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True)
        (DATASET / "labels" / split).mkdir(parents=True)

    for index in range(6):
        split = "train" if index < 4 else "val"
        image = np.full((320, 320, 3), (48, 116, 58), dtype=np.uint8)
        x = 55 + index * 30
        cv2.rectangle(image, (x, 90), (x + 32, 210), (225, 225, 235), -1)
        ball_x, ball_y = 235 - index * 12, 230
        cv2.circle(image, (ball_x, ball_y), 7, (245, 245, 245), -1)
        stem = f"frame-{index:02d}"
        cv2.imwrite(str(DATASET / "images" / split / f"{stem}.jpg"), image)
        player_cx, player_cy = (x + 16) / 320, 150 / 320
        labels = [
            f"0 {player_cx:.6f} {player_cy:.6f} {32/320:.6f} {120/320:.6f}",
            f"1 {ball_x/320:.6f} {ball_y/320:.6f} {14/320:.6f} {14/320:.6f}",
        ]
        (DATASET / "labels" / split / f"{stem}.txt").write_text("\n".join(labels), encoding="utf-8")

    yaml_path = DATASET / "dataset.yaml"
    yaml_path.write_text(
        f"path: {DATASET.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: player\n  1: ball\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    dataset_yaml = create_dataset()
    model = YOLO(str(ROOT / "models" / "yolo26n.pt"))
    started = time.perf_counter()
    result = model.train(
        data=str(dataset_yaml), epochs=1, imgsz=320, batch=1,
        device=0 if torch.cuda.is_available() else "cpu", workers=0,
        project=str(RUNS), name="smoke", exist_ok=True,
        cache=False, plots=False, amp=True, val=False, verbose=False,
    )
    elapsed = time.perf_counter() - started
    checkpoint = Path(result.save_dir) / "weights" / "last.pt"
    report = {
        "status": "passed" if checkpoint.exists() else "failed",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "epochs": 1,
        "training_images": 4,
        "image_size": 320,
        "batch": 1,
        "elapsed_seconds": round(elapsed, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else 0,
        "checkpoint": checkpoint.as_posix(),
    }
    (RUNS / "smoke-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
