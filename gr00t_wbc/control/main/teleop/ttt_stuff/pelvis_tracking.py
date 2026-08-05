"""Track pelvis motion from leg FK while both feet remain fixed."""

import numpy as np
from scipy.spatial.transform import Rotation

from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model


class PelvisTracker:
    def __init__(self):
        self.model = instantiate_g1_robot_model(waist_location="lower_body")
        self.initial = None

    def _foot_from_pelvis(self, q):
        self.model.cache_forward_kinematics(np.asarray(q, dtype=float), auto_clip=False)
        return [
            np.linalg.inv(self.model.frame_placement(frame).homogeneous)
            for frame in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]

    def capture(self, q):
        self.initial = self._foot_from_pelvis(q)

    def delta(self, q):
        """Return current-pelvis from captured-pelvis rotation and translation."""
        if self.initial is None:
            raise RuntimeError("Capture the initial leg state first")
        current = self._foot_from_pelvis(q)
        estimates = [np.linalg.inv(now) @ initial for now, initial in zip(current, self.initial)]
        rotation = Rotation.from_matrix([estimate[:3, :3] for estimate in estimates]).mean().as_matrix()
        translation = np.mean([estimate[:3, 3] for estimate in estimates], axis=0)
        return rotation, translation
