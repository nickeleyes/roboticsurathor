"""Detect the portrait paper, rectify its grid, and classify all nine cells."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

checkpoint = torch.load(Path(__file__).with_name("center_resnet18.pt"),
                        map_location="cpu", weights_only=True)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(checkpoint["class_names"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
transform = transforms.Compose([
    transforms.Resize((checkpoint["input_size"], checkpoint["input_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
OUTPUT_DIR = Path(__file__).with_name("outputs")

# The paper is placed with its 17-inch dimension running away from the robot,
# then rotated back into this landscape coordinate frame by the corner ordering
# below.  Using 100 pixels per inch keeps print measurements easy to translate.
PAPER_WIDTH = 1700
PAPER_HEIGHT = 1100

GRID_CROP = (314, 24, 1081, 1059)


def odd(value):
    return int(round(value)) // 2 * 2 + 1


def detect(frame, save_debug=True):
    if save_debug:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def save(name, image):
        if save_debug:
            cv2.imwrite(str(OUTPUT_DIR / name), image)

    save("color.png", frame)

    scale = min(frame.shape[:2])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(cv2.inRange(gray, 160, 255), (odd(scale * .01),) * 2, 0)
    _, mask = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)
    kernel = np.ones((odd(scale * .012),) * 2, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    save("mask.png", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        at_edge = x <= 2 or y <= 2 or x + width >= frame.shape[1] - 2 or y + height >= frame.shape[0] - 2
        if at_edge or not frame.shape[0] * frame.shape[1] * .01 < area < frame.shape[0] * frame.shape[1] * .8:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for epsilon in np.linspace(.01, .08, 12):
            points = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(points) == 4:
                points = points.reshape(4, 2).astype(np.float32)
                sums, differences = points.sum(1), np.diff(points, axis=1).ravel()
                corners = points[[np.argmin(sums), np.argmin(differences),
                                  np.argmax(sums), np.argmax(differences)]]
                # The physical paper is fixed 90 degrees counterclockwise (green
                # edge at the bottom). Restore its original landscape coordinate
                # frame so vision, localization, and planning share one convention.
                corners = corners[[3, 0, 1, 2]]
                break
        if corners is not None:
            break

    corner_image = frame.copy()
    if corners is None:
        cv2.putText(
            corner_image,
            "NO BOARD-PAPER CORNERS DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        save("corners.png", corner_image)
        save("board_detection.png", corner_image)
        if save_debug:
            print(f"Saved vision debug images to {OUTPUT_DIR}", flush=True)
        raise RuntimeError("No board-paper corners detected")

    cv2.polylines(corner_image, [corners.astype(np.int32)], True, (0, 255, 0), 2)
    for index, point in enumerate(corners.astype(np.int32)):
        cv2.circle(corner_image, tuple(point), 5, (0, 0, 255), -1)
        cv2.putText(
            corner_image,
            str(index),
            tuple(point + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    save("corners.png", corner_image)

    destination = np.array(
        [[0, 0], [PAPER_WIDTH - 1, 0],
         [PAPER_WIDTH - 1, PAPER_HEIGHT - 1], [0, PAPER_HEIGHT - 1]],
        np.float32,
    )
    paper = cv2.warpPerspective(
        frame,
        cv2.getPerspectiveTransform(corners, destination),
        (PAPER_WIDTH, PAPER_HEIGHT),
    )
    grid_x, grid_y, grid_width, grid_height = GRID_CROP
    board = paper[grid_y:grid_y + grid_height, grid_x:grid_x + grid_width]
    save("board_detection.png", board)
    cells = []
    cell_height = board.shape[0] // 3
    cell_width = board.shape[1] // 3
    for row in range(3):
        for column in range(3):
            cell = board[
                row * cell_height:(row + 1) * cell_height,
                column * cell_width:(column + 1) * cell_width,
            ]
            cell = cv2.rotate(cell, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cells.append(transform(Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))))
    with torch.inference_mode():
        label_indices = model(torch.stack(cells)).argmax(1).tolist()
    labels = [checkpoint["class_names"][index] for index in label_indices]
    codes = [{"empty": "0", "green": "1", "white": "2"}[label] for label in labels]

    detected_board = board.copy()
    for index, label in enumerate(labels):
        cv2.putText(
            detected_board,
            f"{index + 1}: {label}",
            ((index % 3) * cell_width + 8, (index // 3) * cell_height + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    save("board_detection.png", detected_board)
    if save_debug:
        print(f"Saved vision debug images to {OUTPUT_DIR}", flush=True)
    return {"board_state": "".join(codes), "corners_uv": corners.tolist()}
