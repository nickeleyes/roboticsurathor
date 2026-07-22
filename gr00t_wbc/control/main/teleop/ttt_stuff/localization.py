"""Convert detected grid-corner centers into the pelvis coordinate frame."""
import cv2
import numpy as np


def localize(detection, radial, intrinsics):
    if radial.shape != (intrinsics["height"], intrinsics["width"]):
        raise ValueError("Radial map and camera intrinsics have different dimensions")
    if any(abs(value) > 1e-9 for value in intrinsics["distortion_coefficients"]):
        raise ValueError("Board localization expects rectified camera images")

    paper = np.asarray(detection["corners_uv"], np.float32)
    canonical = np.array([[0, 0], [999, 0], [999, 773], [0, 773]], np.float32)
    inverse = np.linalg.inv(cv2.getPerspectiveTransform(paper, canonical))
    pixels = cv2.perspectiveTransform(
        np.array([[[270, 160], [730, 160], [270, 620], [730, 620]]], np.float32), inverse
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

    pitch = .8307767239493009
    rotation = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
                         [-np.sin(pitch), 0, np.cos(pitch)]]) @ np.array(
                             [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    measured = np.asarray(camera_points) @ rotation.T + [0.056, 0, 0.504]
    center = measured.mean(0)
    center[2] = np.median(measured[:, 2])
    across = (measured[1] - measured[0] + measured[3] - measured[2]) / 2
    down = (measured[2] - measured[0] + measured[3] - measured[1]) / 2
    across[2] = down[2] = 0
    across /= np.linalg.norm(across)
    down -= across * np.dot(down, across)
    down /= np.linalg.norm(down)
    width, height = .2794 * 460 / 1000, .2159 * 460 / 774
    corners = np.array([
        center - across * width / 2 - down * height / 2,
        center + across * width / 2 - down * height / 2,
        center - across * width / 2 + down * height / 2,
        center + across * width / 2 + down * height / 2,
    ])
    return {"frame_id": "pelvis", "board_state": detection["board_state"],
            "corner_cell_centers": dict(zip(("p1", "p3", "p7", "p9"), corners.tolist()))}
