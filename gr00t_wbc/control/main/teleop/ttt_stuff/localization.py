"""Convert detected grid-corner centers into the pelvis coordinate frame."""
import cv2
import numpy as np

CAMERA_PITCH = 0.8307767239493009
PAPER_WIDTH = 1700
PAPER_HEIGHT = 1100


def neutral_camera_to_pelvis():
    transform = np.eye(4)
    transform[:3, :3] = np.array([
        [np.cos(CAMERA_PITCH), 0, np.sin(CAMERA_PITCH)],
        [0, 1, 0],
        [-np.sin(CAMERA_PITCH), 0, np.cos(CAMERA_PITCH)],
    ]) @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    transform[:3, 3] = [0.056, 0, 0.504]
    return transform


def localize(detection, radial, intrinsics, camera_to_pelvis=None):
    paper = np.asarray(detection["corners_uv"], np.float32)
    canonical = np.array(
        [[0, 0], [PAPER_WIDTH - 1, 0],
         [PAPER_WIDTH - 1, PAPER_HEIGHT - 1], [0, PAPER_HEIGHT - 1]],
        np.float32,
    )
    inverse = np.linalg.inv(cv2.getPerspectiveTransform(paper, canonical))
    pixels = cv2.perspectiveTransform(
        np.array([[[496, 201], [1219, 201], [496, 906], [1219, 906]]], np.float32),
        inverse,
    )[0]

    camera_points = []
    for name, (u, v) in zip(("p1", "p3", "p7", "p9"), pixels):
        x, y = int(round(u)), int(round(v))
        sample = radial[max(0, y - 8):y + 9, max(0, x - 8):x + 9]
        valid = sample[np.isfinite(sample) & (sample > 0)]
        if valid.size < 5:
            raise ValueError(f"No valid depth near {name} at ({u:.1f}, {v:.1f})")
        ray = np.array([(u - intrinsics["cx"]) / intrinsics["fx"],
                        (v - intrinsics["cy"]) / intrinsics["fy"], 1])
        camera_points.append(np.median(valid) * ray / np.linalg.norm(ray))

    if camera_to_pelvis is None:
        camera_to_pelvis = neutral_camera_to_pelvis()
    measured = (
        np.asarray(camera_points) @ camera_to_pelvis[:3, :3].T
        + camera_to_pelvis[:3, 3]
    )
    return detection["board_state"], measured
