"""Track pelvis-frame motion from leg FK while both feet remain fixed."""

import numpy as np
from scipy.spatial.transform import Rotation

from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model


LEFT_FOOT_FRAME = "left_ankle_roll_link"
RIGHT_FOOT_FRAME = "right_ankle_roll_link"


def _matrix(placement):
    transform = np.eye(4)
    transform[:3, :3] = placement.rotation
    transform[:3, 3] = placement.translation
    return transform


class PelvisTracker:
    """Estimate pelvis motion using stationary left and right foot frames."""

    def __init__(self, robot_model=None):
        self.robot_model = robot_model or instantiate_g1_robot_model(
            waist_location="lower_body"
        )
        self.initial = None
        self.final = None

    def pelvis_from_leg_joints(self, q):
        """Return each foot_T_pelvis pose for the supplied full joint state."""
        q = np.asarray(q, dtype=float)
        self.robot_model.cache_forward_kinematics(q, auto_clip=False)
        pelvis_to_left_foot = _matrix(self.robot_model.frame_placement(LEFT_FOOT_FRAME))
        pelvis_to_right_foot = _matrix(self.robot_model.frame_placement(RIGHT_FOOT_FRAME))
        return {
            "left_foot_T_pelvis": np.linalg.inv(pelvis_to_left_foot),
            "right_foot_T_pelvis": np.linalg.inv(pelvis_to_right_foot),
        }

    def capture_initial(self, q):
        self.initial = self.pelvis_from_leg_joints(q)
        self.final = None
        return self.initial

    def capture_final(self, q):
        if self.initial is None:
            raise RuntimeError("Capture the initial leg state before the final leg state")
        self.final = self.pelvis_from_leg_joints(q)
        return self.final

    def pelvis_delta(self):
        """Return the averaged final-pelvis to initial-pelvis transform."""
        if self.initial is None or self.final is None:
            raise RuntimeError("Initial and final leg states are required")

        estimates = {}
        for side in ("left", "right"):
            initial = self.initial[f"{side}_foot_T_pelvis"]
            final = self.final[f"{side}_foot_T_pelvis"]
            estimates[side] = np.linalg.inv(final) @ initial

        delta = np.eye(4)
        delta[:3, 3] = (estimates["left"][:3, 3] + estimates["right"][:3, 3]) / 2
        delta[:3, :3] = Rotation.from_matrix(
            [estimates["left"][:3, :3], estimates["right"][:3, :3]]
        ).mean().as_matrix()
        return delta

    def transform_points(self, points):
        """Transform initial-pelvis points into the final pelvis frame."""
        points = np.asarray(points, dtype=float)
        delta = self.pelvis_delta()
        return points @ delta[:3, :3].T + delta[:3, 3]
