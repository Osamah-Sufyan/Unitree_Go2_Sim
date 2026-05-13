#!/usr/bin/env python3
import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASSES = ["board", "X", "O"]


@dataclass(frozen=True)
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


# Pixel labels for the two seed images. Class ids: board=0, X=1, O=2.
# The board label is the full visible white/tape floor board. Symbols are the
# actual X/O marks, not the grid tape.
SEED_LABELS: dict[str, list[Box]] = {
    "image1.jpg": [
        Box(0, 324, 704, 1482, 1051),
        Box(2, 626, 718, 744, 784),
        Box(2, 901, 721, 1017, 773),
        Box(1, 1127, 710, 1234, 790),
        Box(1, 1164, 774, 1283, 834),
        Box(2, 528, 810, 667, 879),
    ],
    "image2.jpg": [
        Box(0, 280, 704, 1482, 1050),
        Box(1, 632, 750, 721, 800),
        Box(2, 917, 722, 1012, 768),
        Box(1, 871, 777, 987, 837),
        Box(2, 1190, 771, 1296, 819),
        Box(2, 589, 751, 708, 795),
        Box(1, 1215, 828, 1326, 889),
    ],
}


def yolo_line(box: Box, image_width: int, image_height: int) -> str:
    x_center = ((box.x1 + box.x2) / 2.0) / image_width
    y_center = ((box.y1 + box.y2) / 2.0) / image_height
    width = (box.x2 - box.x1) / image_width
    height = (box.y2 - box.y1) / image_height
    return f"{box.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def clip_box(box: Box, image_width: int, image_height: int) -> Box | None:
    x1 = max(0.0, min(float(image_width - 1), box.x1))
    y1 = max(0.0, min(float(image_height - 1), box.y1))
    x2 = max(0.0, min(float(image_width - 1), box.x2))
    y2 = max(0.0, min(float(image_height - 1), box.y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return Box(box.class_id, x1, y1, x2, y2)


def transform_box(box: Box, matrix: np.ndarray, image_width: int, image_height: int) -> Box | None:
    corners = np.array(
        [[[box.x1, box.y1]], [[box.x2, box.y1]], [[box.x2, box.y2]], [[box.x1, box.y2]]],
        dtype=np.float32,
    )
    warped = cv2.transform(corners, matrix).reshape(-1, 2)
    result = Box(box.class_id, warped[:, 0].min(), warped[:, 1].min(), warped[:, 0].max(), warped[:, 1].max())
    return clip_box(result, image_width, image_height)


def augment(image: np.ndarray, boxes: list[Box], rng: random.Random) -> tuple[np.ndarray, list[Box]]:
    height, width = image.shape[:2]
    angle = rng.uniform(-4.0, 4.0)
    scale = rng.uniform(0.92, 1.08)
    tx = rng.uniform(-0.035, 0.035) * width
    ty = rng.uniform(-0.035, 0.035) * height

    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[:, 2] += (tx, ty)
    output = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    alpha = rng.uniform(0.82, 1.18)
    beta = rng.uniform(-18, 18)
    output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)

    if rng.random() < 0.35:
        output = cv2.GaussianBlur(output, (3, 3), 0)

    noise_sigma = rng.uniform(0.0, 5.0)
    if noise_sigma > 0.5:
        noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
        noise_image = noise_rng.normal(0.0, noise_sigma, output.shape).astype(np.float32)
        output = np.clip(output.astype(np.float32) + noise_image, 0, 255).astype(np.uint8)

    transformed_boxes = [box for box in (transform_box(box, matrix, width, height) for box in boxes) if box]
    return output, transformed_boxes


def write_sample(image: np.ndarray, boxes: list[Box], image_path: Path, label_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), image)
    height, width = image.shape[:2]
    label_path.write_text("\n".join(yolo_line(box, width, height) for box in boxes) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an augmented YOLO dataset for tic-tac-toe board detection.")
    parser.add_argument("--output", default="yolo_dataset", help="Dataset output directory")
    parser.add_argument("--augmentations", type=int, default=80, help="Augmented images per seed image")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    root = Path(args.output)
    if root.exists():
        shutil.rmtree(root)

    rng = random.Random(args.seed)
    split_paths = {
        "train": (root / "images" / "train", root / "labels" / "train"),
        "val": (root / "images" / "val", root / "labels" / "val"),
    }

    all_samples: list[tuple[str, np.ndarray, list[Box]]] = []
    for image_name, boxes in SEED_LABELS.items():
        image = cv2.imread(image_name)
        if image is None:
            raise FileNotFoundError(image_name)
        all_samples.append((Path(image_name).stem, image, boxes))
        for index in range(args.augmentations):
            augmented, augmented_boxes = augment(image, boxes, rng)
            all_samples.append((f"{Path(image_name).stem}_aug_{index:03d}", augmented, augmented_boxes))

    rng.shuffle(all_samples)
    val_count = max(4, int(len(all_samples) * 0.18))
    for index, (stem, image, boxes) in enumerate(all_samples):
        split = "val" if index < val_count else "train"
        image_dir, label_dir = split_paths[split]
        write_sample(image, boxes, image_dir / f"{stem}.jpg", label_dir / f"{stem}.txt")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CLASSES)],
                "",
            ]
        )
    )
    print(f"Wrote {len(all_samples) - val_count} train and {val_count} val images to {root}")
    print(f"Dataset config: {data_yaml}")


if __name__ == "__main__":
    main()
