from pathlib import Path
import runpy

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
VISION = ROOT / "gr00t_wbc/control/main/teleop/ttt_stuff/vision.py"
LOCALIZATION = ROOT / "gr00t_wbc/control/main/teleop/ttt_stuff/localization.py"
PLANNER = ROOT / "gr00t_wbc/control/main/teleop/ttt_stuff/planner.py"


def test_vision_normalizes_fixed_counterclockwise_portrait_board():
    source = VISION.read_text()
    assert "corners = corners[[3, 0, 1, 2]]" in source


def test_canonical_ghost_cache_maps_to_physical_bottom_and_top():
    board_planner = runpy.run_path(str(PLANNER))["BoardPlanner"]()

    # Canonical p1/p3/p7/p9 after the physical paper has been rotated CCW:
    # canonical left is physical bottom and canonical right is physical top.
    corners = np.array(
        [
            [0.0, 1.0, 0.0],  # p1
            [0.0, 0.0, 0.0],  # p3
            [1.0, 1.0, 0.0],  # p7
            [1.0, 0.0, 0.0],  # p9
        ]
    )
    points, _ = board_planner.points(corners)

    green_y = [points[f"green_square{i}"][1] for i in range(1, 6)]
    white_y = [points[f"white_square{i}"][1] for i in range(1, 6)]
    assert min(green_y) > 1.0
    assert max(white_y) < 0.0


def test_planner_preserves_tilted_board_coordinates_and_uses_its_normal():
    board_planner = runpy.run_path(str(PLANNER))["BoardPlanner"]()
    corners = np.array(
        [
            [0.4, 0.2, -0.03],
            [0.4, 0.0, -0.01],
            [0.2, 0.2, -0.07],
            [0.2, 0.0, -0.05],
        ]
    )

    points, rotation = board_planner.points(corners)

    for index, corner in zip((1, 3, 7, 9), corners):
        np.testing.assert_allclose(points[f"pos{index}"], corner)

    position, _ = board_planner.target(corners, "pos5", 0.06)
    board_center = corners.mean(axis=0)
    board_normal = -rotation[:, 1]
    np.testing.assert_allclose(position, board_center + 0.06 * board_normal)


def test_localization_returns_transformed_depth_samples_without_regularization():
    localize = runpy.run_path(str(LOCALIZATION))["localize"]
    detection = {
        "board_state": "000000000",
        "corners_uv": [[0, 0], [1699, 0], [1699, 1099], [0, 1099]],
    }
    radial = np.full((1100, 1700), 2.0)
    intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 500.0, "cy": 387.0}
    camera_to_pelvis = np.eye(4)
    camera_to_pelvis[:3, 3] = [0.1, -0.2, 0.3]

    board, corners = localize(detection, radial, intrinsics, camera_to_pelvis)

    pixels = np.array([[496, 201], [1219, 201], [496, 906], [1219, 906]])
    rays = np.column_stack(
        (
            (pixels[:, 0] - intrinsics["cx"]) / intrinsics["fx"],
            (pixels[:, 1] - intrinsics["cy"]) / intrinsics["fy"],
            np.ones(4),
        )
    )
    expected = 2.0 * rays / np.linalg.norm(rays, axis=1, keepdims=True)
    expected += camera_to_pelvis[:3, 3]
    assert board == detection["board_state"]
    np.testing.assert_allclose(corners, expected)
