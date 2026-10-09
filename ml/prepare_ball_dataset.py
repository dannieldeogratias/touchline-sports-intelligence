import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/football-image-dataset"
TARGET = ROOT / "data/processed/football-ball-only"


def main() -> None:
    total_images = total_boxes = positive_images = 0
    for split in ("train", "val"):
        image_target = TARGET / "images" / split
        label_target = TARGET / "labels" / split
        image_target.mkdir(parents=True, exist_ok=True)
        label_target.mkdir(parents=True, exist_ok=True)
        for image in (SOURCE / "images" / split).iterdir():
            if not image.is_file():
                continue
            linked = image_target / image.name
            if not linked.exists():
                os.link(image, linked)
            source_label = SOURCE / "labels" / split / f"{image.stem}.txt"
            ball_lines = [line for line in source_label.read_text(encoding="utf-8").splitlines()
                          if line.strip().split()[0] == "0"] if source_label.exists() else []
            (label_target / f"{image.stem}.txt").write_text("\n".join(ball_lines), encoding="utf-8")
            total_images += 1
            total_boxes += len(ball_lines)
            positive_images += bool(ball_lines)
    (TARGET / "data.yaml").write_text(
        f"path: {TARGET.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n",
        encoding="utf-8",
    )
    print({"images": total_images, "positive_images": positive_images, "ball_boxes": total_boxes,
           "negative_images": total_images - positive_images, "dataset": str(TARGET / 'data.yaml')})


if __name__ == "__main__":
    main()
