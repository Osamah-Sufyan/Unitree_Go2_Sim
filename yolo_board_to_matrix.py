#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_NAMES = {0: "board", 1: "X", 2: "O"}


def pick_best_board(detections: list[dict]) -> dict:
    boards = [d for d in detections if d["class_id"] == 0]
    if not boards:
        raise RuntimeError("YOLO did not detect a board")
    return max(boards, key=lambda d: d["confidence"])


def detections_to_matrix(detections: list[dict], confidence: float) -> list[list[str]]:
    board = pick_best_board(detections)
    x1, y1, x2, y2 = board["box"]
    board_w = x2 - x1
    board_h = y2 - y1
    matrix = [["" for _ in range(3)] for _ in range(3)]
    scores = [[0.0 for _ in range(3)] for _ in range(3)]

    for detection in detections:
        if detection["class_id"] == 0 or detection["confidence"] < confidence:
            continue
        sx1, sy1, sx2, sy2 = detection["box"]
        cx = (sx1 + sx2) / 2.0
        cy = (sy1 + sy2) / 2.0
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue

        col = min(2, max(0, int(((cx - x1) / board_w) * 3)))
        row = min(2, max(0, int(((cy - y1) / board_h) * 3)))
        if detection["confidence"] > scores[row][col]:
            matrix[row][col] = CLASS_NAMES[detection["class_id"]]
            scores[row][col] = detection["confidence"]

    return matrix


def draw_debug(image_path: str, detections: list[dict], matrix: list[list[str]], output_path: str) -> None:
    image = cv2.imread(image_path)
    for detection in detections:
        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        color = (255, 0, 0) if detection["class_id"] == 0 else (0, 220, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[detection['class_id']]} {detection['confidence']:.2f}"
        cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    board = pick_best_board(detections)
    x1, y1, x2, y2 = [int(v) for v in board["box"]]
    for i in (1, 2):
        px = int(x1 + (x2 - x1) * i / 3.0)
        py = int(y1 + (y2 - y1) * i / 3.0)
        cv2.line(image, (px, y1), (px, y2), (0, 255, 255), 2)
        cv2.line(image, (x1, py), (x2, py), (0, 255, 255), 2)

    for row in range(3):
        for col in range(3):
            cv2.putText(
                image,
                matrix[row][col] or "-",
                (int(x1 + (col + 0.45) * (x2 - x1) / 3.0), int(y1 + (row + 0.55) * (y2 - y1) / 3.0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
    cv2.imwrite(output_path, image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained YOLO model and return a tic-tac-toe matrix.")
    parser.add_argument("weights", help="Path to trained YOLO weights, usually runs/tictactoe_yolo/weights/best.pt")
    parser.add_argument("image")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--debug", help="Optional annotated output image path")
    args = parser.parse_args()

    model = YOLO(args.weights)
    result = model(args.image, conf=args.conf, verbose=False)[0]
    detections = []
    for box in result.boxes:
        detections.append(
            {
                "class_id": int(box.cls.item()),
                "class_name": CLASS_NAMES.get(int(box.cls.item()), str(int(box.cls.item()))),
                "confidence": float(box.conf.item()),
                "box": [float(v) for v in box.xyxy[0].tolist()],
            }
        )

    matrix = detections_to_matrix(detections, args.conf)
    output = {"matrix": matrix, "detections": detections}
    print(json.dumps(output, indent=2))

    if args.debug:
        draw_debug(args.image, detections, matrix, args.debug)


if __name__ == "__main__":
    main()
