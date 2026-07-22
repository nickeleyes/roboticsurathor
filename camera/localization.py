"""Localize detected corner-cell centers from one saved synchronized capture."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAPTURE = ROOT / "camera_captures"


def localize(detection=None, capture=CAPTURE):
    if detection is None:
        detection = json.loads((capture / "board_detection/result.json").read_text())
    intrinsics = json.loads((capture / "camera_intrinsics.json").read_text())
    radial = np.load(capture / "radial_distance_m.npy")
    if radial.shape != (intrinsics["height"], intrinsics["width"]):
        raise ValueError("Radial map and camera intrinsics have different dimensions")
    if any(abs(value) > 1e-9 for value in intrinsics["distortion_coefficients"]):
        raise ValueError("This temporary localizer expects rectified camera images")

    paper = np.asarray(detection["corners_uv"], np.float32)
    canonical = np.array([[0, 0], [999, 0], [999, 773], [0, 773]], np.float32)
    inverse_homography = np.linalg.inv(cv2.getPerspectiveTransform(paper, canonical))
    centers = np.array([[270, 160], [730, 160], [270, 620], [730, 620]], np.float32)
    pixels = cv2.perspectiveTransform(centers[None], inverse_homography)[0]

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    camera_points, measurements = [], []
    for name, (u, v) in zip(("p1", "p3", "p7", "p9"), pixels):
        x, y = int(round(u)), int(round(v))
        sample = radial[max(0, y - 8):y + 9, max(0, x - 8):x + 9]
        valid = sample[np.isfinite(sample) & (sample > 0)]
        if valid.size < 5:
            raise ValueError(f"No valid depth near {name} at ({u:.1f}, {v:.1f})")
        distance = float(np.median(valid))
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        point = distance * ray / np.linalg.norm(ray)
        camera_points.append(point)
        measurements.append({
            "name": name,
            "pixel_uv": [float(u), float(v)],
            "radial_distance_m": distance,
            "horizontal_angle_rad": float(np.arctan2(ray[0], 1.0)),
            "vertical_angle_rad": float(np.arctan2(ray[1], 1.0)),
            "camera_xyz_m": point.tolist(),
        })

    pitch = 0.8307767239493009
    rotation = np.array([
        [np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ]) @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    translation = np.array([0.056, 0.0, 0.504])
    measured = np.asarray(camera_points) @ rotation.T + translation

    center = measured.mean(axis=0)
    center[2] = np.median(measured[:, 2])
    across = (measured[1] - measured[0] + measured[3] - measured[2]) / 2
    down = (measured[2] - measured[0] + measured[3] - measured[1]) / 2
    across[2] = down[2] = 0
    across /= np.linalg.norm(across)
    down -= across * np.dot(down, across)
    down /= np.linalg.norm(down)
    width = 0.2794 * 460 / 1000      # 11-inch landscape paper width
    height = 0.2159 * 460 / 774      # 8.5-inch landscape paper height
    regularized = np.array([
        center - across * width / 2 - down * height / 2,
        center + across * width / 2 - down * height / 2,
        center - across * width / 2 + down * height / 2,
        center + across * width / 2 + down * height / 2,
    ])
    result = {
        "frame_id": "pelvis",
        "board_state": detection["board_state"],
        "camera_to_pelvis": {
            "capture_pose": "neutral_waist",
            "translation_m": translation.tolist(),
            "rotation": rotation.tolist(),
        },
        "measurements": measurements,
        "measured_pelvis_xyz_m": dict(zip(("p1", "p3", "p7", "p9"), measured.tolist())),
        "corner_cell_centers": dict(zip(("p1", "p3", "p7", "p9"), regularized.tolist())),
    }
    output = capture / "board_localization.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    for name, point in result["corner_cell_centers"].items():
        print(f"{name}: {np.round(point, 4)} m")
    return result


def main():
    localize()


if __name__ == "__main__":
    main()
