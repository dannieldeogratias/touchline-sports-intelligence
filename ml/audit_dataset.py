import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2


def perceptual_hash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    resized = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    mean = float(resized.mean())
    value = 0
    for pixel in resized.flat:
        value = (value << 1) | int(pixel >= mean)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/dataset-audit.json"))
    args = parser.parse_args()

    class_counts: Counter[int] = Counter()
    split_counts = {}
    invalid_labels = []
    missing_pairs = []
    image_hashes = {}
    content_hashes = {}

    for split in ("train", "val"):
        images = sorted((args.dataset / "images" / split).glob("*.jpg"))
        labels = {path.stem: path for path in (args.dataset / "labels" / split).glob("*.txt")}
        split_counts[split] = len(images)
        for image_path in images:
            label_path = labels.pop(image_path.stem, None)
            if label_path is None:
                missing_pairs.append(str(image_path))
                continue
            content_hashes[(split, image_path.name)] = hashlib.sha256(image_path.read_bytes()).hexdigest()
            image_hashes[(split, image_path.name)] = perceptual_hash(image_path)
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    class_id = int(parts[0])
                    coordinates = [float(value) for value in parts[1:]]
                    valid = len(coordinates) == 4 and class_id in {0, 1, 2} and all(0 <= value <= 1 for value in coordinates)
                except (ValueError, IndexError):
                    valid = False
                    class_id = -1
                if not valid:
                    invalid_labels.append(f"{label_path}:{line_number}")
                else:
                    class_counts[class_id] += 1
        missing_pairs.extend(str(path) for path in labels.values())

    train_items = [(name, value) for (split, name), value in image_hashes.items() if split == "train"]
    val_items = [(name, value) for (split, name), value in image_hashes.items() if split == "val"]
    near_cross_split = []
    for train_name, train_hash in train_items:
        for val_name, val_hash in val_items:
            distance = (train_hash ^ val_hash).bit_count()
            if distance <= 5:
                near_cross_split.append({"train": train_name, "val": val_name, "distance": distance})

    train_sha = {value: name for (split, name), value in content_hashes.items() if split == "train"}
    exact_cross_split = [
        {"train": train_sha[value], "val": name}
        for (split, name), value in content_hashes.items() if split == "val" and value in train_sha
    ]
    report = {
        "dataset": str(args.dataset),
        "images": split_counts,
        "boxes": {"ball": class_counts[0], "player": class_counts[1], "referee": class_counts[2]},
        "missing_pairs": missing_pairs,
        "invalid_labels": invalid_labels,
        "exact_cross_split_duplicates": exact_cross_split,
        "near_cross_split_pairs_at_hash_distance_5": near_cross_split,
        "split_provenance_warning": "Filenames contain no match identifier, so match-grouped split isolation cannot be verified.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "near_cross_split_pairs_at_hash_distance_5": len(near_cross_split)}, indent=2))


if __name__ == "__main__":
    main()
