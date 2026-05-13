#!/usr/bin/env python3
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on the augmented tic-tac-toe dataset.")
    parser.add_argument("--data", default="yolo_dataset/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO checkpoint, e.g. yolo11n.pt or yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="tictactoe_yolo")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=20,
        degrees=2.0,
        translate=0.04,
        scale=0.08,
        fliplr=0.0,
        mosaic=0.0,
        verbose=True,
    )


if __name__ == "__main__":
    main()
