"""Run board detection and piece classification on a saved ROS RGB image."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = ROOT / "camera_captures/color_rgb.png"
DEFAULT_MODEL = ROOT / "camera/models/center_resnet18.pt"
DEFAULT_OUTPUT = ROOT / "camera_captures/board_detection"
GRID_CROP = (155, 45, 690, 690)
LABEL_TO_CODE = {"empty": "0", "green": "1", "white": "2"}


def odd_kernel(value):
    value = max(3, int(round(value)))
    return value if value % 2 else value + 1


def order_corners(points):
    ordered = np.zeros((4, 2), np.float32)
    totals = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0], ordered[2] = points[np.argmin(totals)], points[np.argmax(totals)]
    ordered[1], ordered[3] = points[np.argmin(differences)], points[np.argmax(differences)]
    return ordered  # top-left, top-right, bottom-right, bottom-left


def find_board_corners(frame):
    scale = min(frame.shape[:2])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 160, 255)
    mask = cv2.GaussianBlur(mask, (odd_kernel(scale * 0.01),) * 2, 0)
    _, mask = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)
    kernel = np.ones((odd_kernel(scale * 0.012),) * 2, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        at_edge = x <= 2 or y <= 2 or x + width >= frame.shape[1] - 2 or y + height >= frame.shape[0] - 2
        if at_edge or not frame_area * 0.01 < area < frame_area * 0.80:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for epsilon in np.linspace(0.01, 0.08, 12):
            corners = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(corners) == 4:
                return order_corners(corners.reshape(4, 2).astype(np.float32))
    return None


def warp_board(frame, corners):
    destination = np.array([[0, 0], [999, 0], [999, 773], [0, 773]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(frame, matrix, (1000, 774))


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    names, size = checkpoint["class_names"], checkpoint["input_size"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, names, transform


def detect(image=DEFAULT_IMAGE, model_path=DEFAULT_MODEL, output=DEFAULT_OUTPUT):
    frame = cv2.imread(str(image))
    if frame is None:
        raise FileNotFoundError(f"Could not read {image}")
    corners = find_board_corners(frame)
    if corners is None:
        raise RuntimeError("No board-paper corners detected")
    board_image = warp_board(frame, corners)
    x, y, width, height = GRID_CROP
    grid = board_image[y:y + height, x:x + width]
    model, names, transform = load_model(model_path)
    output.mkdir(parents=True, exist_ok=True)
    squares_dir = output / "squares"
    squares_dir.mkdir(exist_ok=True)

    predictions, codes = [], []
    annotated_grid = grid.copy()
    cell_h, cell_w = grid.shape[0] // 3, grid.shape[1] // 3
    for row in range(3):
        for column in range(3):
            number = row * 3 + column + 1
            cell = grid[row * cell_h:(row + 1) * cell_h, column * cell_w:(column + 1) * cell_w]
            tensor = transform(Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))).unsqueeze(0)
            with torch.inference_mode():
                probabilities = torch.softmax(model(tensor), dim=1)[0]
            index = int(probabilities.argmax())
            label, confidence = names[index], float(probabilities[index])
            predictions.append({"cell": number, "label": label, "confidence": confidence})
            codes.append(LABEL_TO_CODE[label])
            cv2.imwrite(str(squares_dir / f"square_{number}.png"), cell)
            origin = (column * cell_w + 6, row * cell_h + 24)
            cv2.putText(annotated_grid, f"{number}:{label} {confidence:.2f}", origin,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    annotated = frame.copy()
    cv2.polylines(annotated, [corners.astype(np.int32)], True, (0, 255, 0), 2)
    for index, point in enumerate(corners.astype(int)):
        cv2.circle(annotated, tuple(point), 4, (0, 0, 255), -1)
        cv2.putText(annotated, str(index), tuple(point + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

    board_state = "".join(codes)
    cv2.imwrite(str(output / "corners_detected.png"), annotated)
    cv2.imwrite(str(output / "3x3board.png"), grid)
    cv2.imwrite(str(output / "3x3board_detected.png"), annotated_grid)
    result = {"board_state": board_state, "corners_uv": corners.tolist(), "cells": predictions}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(f"Board state: {board_state} (0=empty, 1=green, 2=white)")
    print(f"Output: {output.resolve()}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    detect(args.image, args.model, args.output)


if __name__ == "__main__":
    main()
