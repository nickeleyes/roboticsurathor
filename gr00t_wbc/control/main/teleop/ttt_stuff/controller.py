import numpy as np

from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from gr00t_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK

# Opening and grabbing change only the middle finger. The index stays extended and
# the thumb stays in the tuned static pose.
RIGHT_HAND_POSES = {
    "right_hand_thumb_0_joint": (-0.45, -0.45),
    "right_hand_thumb_1_joint": (-0.45, -0.45),
    "right_hand_thumb_2_joint": (-0.7, -0.7),
    "right_hand_index_0_joint": (0.0, 0.0),
    "right_hand_index_1_joint": (0.0, 0.0),
    "right_hand_middle_0_joint": (0.0, 0.8),
    "right_hand_middle_1_joint": (0.0, 0.85),
}
LEFT_ARM_AT_SIDE = np.array([0.0, np.deg2rad(30.0), 0.0, np.pi / 2, 0.0, 0.0, 0.0])


class Controller:
    def __init__(self):
        self.model = instantiate_g1_robot_model(waist_location="upper_body")
        waist = self.model.dof_index("waist_yaw_joint")
        self.waist_roll_pitch = [
            self.model.dof_index(name)
            for name in ("waist_roll_joint", "waist_pitch_joint")
        ]
        self.site = "right_hand_grip_center"
        self.model.supplemental_info.hand_frame_names["right"] = self.site

        left_hand_ik, right_hand_ik = instantiate_g1_hand_ik_solver()
        self.solver = TeleopRetargetingIK(
            self.model,
            left_hand_ik,
            right_hand_ik,
            body_active_joint_groups=["right_arm", "waist_yaw_only"],
        )
        self.solver.in_warmup = False
        self.solver.body_ik_solver.num_step_per_frame = 5
        self.solver.body_ik_solver.update_weights(
            {
                self.site: {
                    "position_cost": 100.0,
                    "orientation_cost": np.array([15.0, 15.0, 15.0]),
                }
            }
        )
        upper_body = self.model.get_joint_group_indices("upper_body")
        local_index = {joint: index for index, joint in enumerate(upper_body)}
        self.upper_joint_index = {
            name: local_index[self.model.dof_index(name)]
            for name in RIGHT_HAND_POSES
        }

        def group(name):
            return [local_index[joint] for joint in self.model.get_joint_group_indices(name)]

        self.waist_yaw = local_index[waist]
        self.left_arm = group("left_arm")
        self.right_arm = group("right_arm")
        self.left_hand = group("left_hand")
        self.right_hand = group("right_hand")
        self.joints = self.model.default_body_pose[upper_body].copy()
        self._apply_fixed_joints(self.joints)

    def _apply_fixed_joints(self, joints, grabbing=False):
        joints[self.left_arm] = LEFT_ARM_AT_SIDE
        joints[self.left_hand] = 0.0
        for joint_name, poses in RIGHT_HAND_POSES.items():
            joints[self.upper_joint_index[joint_name]] = poses[grabbing]

    def ik(self, position, rotation, q, grabbing=False):
        target = np.eye(4)
        target[:3, 3] = np.array(position, dtype=float)
        target[:3, :3] = np.array(rotation, dtype=float)

        q = np.asarray(q, dtype=float)
        self.model.cache_forward_kinematics(q, auto_clip=False)
        actual_torso = self.model.frame_placement("torso_link").homogeneous.copy()
        q = q.copy()
        q[self.waist_roll_pitch] = 0.0
        self.model.cache_forward_kinematics(q, auto_clip=False)
        zero_torso = self.model.frame_placement("torso_link").homogeneous
        target = zero_torso @ np.linalg.inv(actual_torso) @ target

        self.solver.set_goal({"body_data": {self.site: target}, "left_hand_data": None, "right_hand_data": None})
        self.joints = self.solver.get_action().copy()
        self._apply_fixed_joints(self.joints, grabbing)
        return self.joints
