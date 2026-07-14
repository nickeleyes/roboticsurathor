from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from gr00t_wbc.control.main.teleop.board_camera_geometry import (
    board_geometry_from_camera_depth,
)


@dataclass(frozen=True)
class Waypoint:
    name: str
    position: np.ndarray
    rotation: np.ndarray
    hand: str
    duration: float


class BoardPlanner:
    """Turn a board move into a short Cartesian pick-and-place plan."""

    def __init__(
        self,
        hover_height: float = 0.16,
        grasp_height: float = 0.025,
        supply_offset: float = 0.13,
        supply_spacing: float = 0.045,
    ):
        self.hover_height = hover_height
        self.grasp_height = grasp_height
        self.supply_offset = supply_offset
        self.supply_spacing = supply_spacing

    def plan(self, board: str, target_cell: int, floating_base_pose) -> list[Waypoint]:
        self._validate(board, target_cell)
        pose = np.asarray(floating_base_pose, dtype=float)
        if pose.shape != (7,):
            raise ValueError(f"floating_base_pose must have shape (7,), got {pose.shape}")
        quaternion = pose[[4, 5, 6, 3]]
        pelvis_rotation = (
            np.eye(3) if np.linalg.norm(quaternion) < 1e-6 else R.from_quat(quaternion).as_matrix()
        )
        # Real hardware reports zero translation when global odometry is not
        # available; use the model's calibrated pelvis origin in that case.
        pelvis_position = None if np.allclose(pose[:3], 0.0) else pose[:3]
        cells, palm_rotation = board_geometry_from_camera_depth(
            pelvis_position, pelvis_rotation
        )

        normal = -palm_rotation[:, 1]
        toward_front = self._unit(cells["1"] - cells["7"])
        toward_left = self._unit(cells["1"] - cells["3"])
        piece_number = board.count("1")
        supply = (
            cells["5"]
            + toward_left * self.supply_offset
            + toward_front * (2 - piece_number) * self.supply_spacing
        )
        target = cells[str(target_cell)]
        center = cells["5"]

        def at(name, point, height, hand, duration):
            return Waypoint(
                name=name,
                position=point + normal * height,
                rotation=palm_rotation,
                hand=hand,
                duration=duration,
            )

        return [
            at("board center", center, self.hover_height, "open", 1.5),
            at("supply hover", supply, self.hover_height, "open", 1.5),
            at("supply grasp", supply, self.grasp_height, "open", 1.0),
            at("close hand", supply, self.grasp_height, "closed", 0.7),
            at("lift piece", supply, self.hover_height, "closed", 1.2),
            at("target hover", target, self.hover_height, "closed", 1.5),
            at("place piece", target, self.grasp_height, "closed", 1.0),
            at("release piece", target, self.grasp_height, "open", 0.7),
            at("retreat", target, self.hover_height, "open", 1.2),
            at("return to center", center, self.hover_height, "open", 1.5),
        ]

    @staticmethod
    def _validate(board: str, target_cell: int):
        if len(board) != 9 or any(value not in "012" for value in board):
            raise ValueError("board must contain exactly nine characters from 0, 1, and 2")
        if target_cell not in range(1, 10):
            raise ValueError("target_cell must be from 1 to 9")
        if board[target_cell - 1] != "0":
            raise ValueError(f"board cell {target_cell} is occupied")
        if board.count("1") >= 5:
            raise ValueError("no green supply pieces remain")

    @staticmethod
    def _unit(vector):
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            raise ValueError("board geometry contains coincident points")
        return vector / norm
