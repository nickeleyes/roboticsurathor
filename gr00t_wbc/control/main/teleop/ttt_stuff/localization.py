"""Convert detected grid-corner centers into the pelvis coordinate frame."""
import cv2
import numpy as np

CAMERA_PITCH = 0.8307767239493009


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

    if camera_to_pelvis is None:
        camera_to_pelvis = neutral_camera_to_pelvis()
    measured = (
        np.asarray(camera_points) @ camera_to_pelvis[:3, :3].T
        + camera_to_pelvis[:3, 3]
    )
    vertical_drop = camera_to_pelvis[2, 3] - measured[:, 2]
    drops = ", ".join(
        f"{name}={drop:.4f} m" for name, drop in zip(("p1", "p3", "p7", "p9"), vertical_drop)
    )
    print(
        f"Camera-to-board vertical drop (pelvis Z): {drops}; median={np.median(vertical_drop):.4f} m",
        flush=True,
    )
    board_z = np.median(measured[:, 2])
    directions = np.asarray(camera_points)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions @ camera_to_pelvis[:3, :3].T
    distances = (board_z - camera_to_pelvis[2, 3]) / directions[:, 2]
    measured = camera_to_pelvis[:3, 3] + distances[:, None] * directions

    center = measured.mean(0)
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
    return detection["board_state"], corners
