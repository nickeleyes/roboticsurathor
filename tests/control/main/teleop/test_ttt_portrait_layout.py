from pathlib import Path
import runpy

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
VISION = ROOT / "gr00t_wbc/control/main/teleop/ttt_stuff/vision.py"
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
