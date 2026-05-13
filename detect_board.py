#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


Cell = tuple[int, int, int, int]


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    by_y = points[np.argsort(points[:, 1])]
    top = by_y[:2]
    bottom = by_y[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _find_board_quad(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # White board panels: low saturation, high value. Restricting to the lower
    # part of the image avoids bright cabinets/lights in the background.
    white = cv2.inRange(hsv, np.array([0, 0, 190]), np.array([180, 100, 255]))
    white[: int(height * 0.60), :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    board_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area > width * height * 0.002 and y > int(height * 0.60) and x > int(width * 0.12):
            board_contours.append(contour)

    if not board_contours:
        raise RuntimeError("could not find the white tic-tac-toe board")

    points = np.vstack(board_contours)
    hull = cv2.convexHull(points)
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True).reshape(-1, 2)

    if len(approx) < 4:
        x, y, w, h = cv2.boundingRect(points)
        approx = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

    # The board is a perspective quadrilateral. In these camera views the two
    # top corners are the two highest hull points, while the two bottom corners
    # are the leftmost/rightmost points from the remaining lower hull.
    by_y = approx[np.argsort(approx[:, 1])]
    top = by_y[:2]
    lower = by_y[2:]
    if len(lower) < 2:
        lower = approx

    tl, tr = top[np.argsort(top[:, 0])]
    bl = lower[np.argmin(lower[:, 0])]
    br = lower[np.argmax(lower[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _classify_cell(image: np.ndarray, cell: Cell) -> str:
    x, y, w, h = cell
    mx = int(w * 0.16)
    my = int(h * 0.18)
    roi = image[y + my : y + h - my, x + mx : x + w - mx]
    if roi.size == 0:
        return ""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Black symbols on a white board. Otsu handles lighting differences across
    # the room, then morphology removes camera noise and thin panel seams.
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    mark_ratio = cv2.countNonZero(dark) / float(dark.shape[0] * dark.shape[1])
    if mark_ratio < 0.035:
        return ""

    contours, hierarchy = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < dark.shape[0] * dark.shape[1] * 0.015:
                continue
            has_child = hierarchy[0][index][2] != -1
            if has_child:
                return "O"

    return "X"


def detect_board(image_path: str | Path) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    board_quad = _find_board_quad(image)
    board_size = 900
    destination = np.array(
        [[0, 0], [board_size - 1, 0], [board_size - 1, board_size - 1], [0, board_size - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(board_quad, destination)
    board = cv2.warpPerspective(image, transform, (board_size, board_size))

    cell_size = board_size // 3
    grid = [
        [(col * cell_size, row * cell_size, cell_size, cell_size) for col in range(3)]
        for row in range(3)
    ]
    matrix = [[_classify_cell(board, cell) for cell in row] for row in grid]

    xs = board_quad[:, 0]
    ys = board_quad[:, 1]

    return {
        "matrix": matrix,
        "board_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "board_quad": [[int(x), int(y)] for x, y in board_quad],
        "cells": [[[int(v) for v in cell] for cell in row] for row in grid],
    }


def save_debug(image_path: str | Path, result: dict, output_path: str | Path) -> None:
    image = cv2.imread(str(image_path))
    quad = np.array(result["board_quad"], dtype=np.int32)
    cv2.polylines(image, [quad], True, (255, 0, 0), 3)
    for start, end in [((1 / 3, 0), (1 / 3, 1)), ((2 / 3, 0), (2 / 3, 1)), ((0, 1 / 3), (1, 1 / 3)), ((0, 2 / 3), (1, 2 / 3))]:
        p1 = _quad_point(quad.astype(np.float32), *start)
        p2 = _quad_point(quad.astype(np.float32), *end)
        cv2.line(image, p1, p2, (0, 220, 0), 2)

    for row_index in range(3):
        for col_index in range(3):
            point = _quad_point(quad.astype(np.float32), (col_index + 0.5) / 3, (row_index + 0.5) / 3)
            cv2.putText(
                image,
                result["matrix"][row_index][col_index] or "-",
                point,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
    x1, y1, x2, y2 = result["board_bbox"]
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.imwrite(str(output_path), image)


def _quad_point(quad: np.ndarray, u: float, v: float) -> tuple[int, int]:
    top = quad[0] * (1 - u) + quad[1] * u
    bottom = quad[3] * (1 - u) + quad[2] * u
    point = top * (1 - v) + bottom * v
    return int(point[0]), int(point[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect a tic-tac-toe board state from an image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--debug", help="Optional path for an annotated output image")
    args = parser.parse_args()

    result = detect_board(args.image)
    print(json.dumps(result, indent=2))

    if args.debug:
        save_debug(args.image, result, args.debug)


if __name__ == "__main__":
    main()
