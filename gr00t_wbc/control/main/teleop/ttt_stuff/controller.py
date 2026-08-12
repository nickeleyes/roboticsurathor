import numpy as np

from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from gr00t_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK

# Opening and grabbing change only the middle finger. The index stays extended and
# the thumb stays in the tuned static pose.
HAND_POSES = {
    "thumb_0": (-0.45, -0.45),
    "thumb_1": (0.5, -0.60),
    "thumb_2": (-0.5, -0.8),
    "index_0": (0.0, 0.0),
    "index_1": (0.0, 0.0),
    "middle_0": (np.pi / 2 * (75/90) , np.pi / 2 * (75/90)),
    "middle_1": (0.0, 0.0),
}


class Controller:
    def __init__(self, arm_side="right"):
        self.arm_side = arm_side
        self.other_side = "left" if arm_side == "right" else "right"
        mirror = 1 if arm_side == "right" else -1
        self.model = instantiate_g1_robot_model(waist_location="upper_body")
        waist = self.model.dof_index("waist_yaw_joint")
        self.waist_roll_pitch = [
            self.model.dof_index(name)
            for name in ("waist_roll_joint", "waist_pitch_joint")
        ]
        self.site = f"{arm_side}_hand_grip_center"
        self.model.supplemental_info.hand_frame_names[arm_side] = self.site

        left_hand_ik, right_hand_ik = instantiate_g1_hand_ik_solver()
        self.solver = TeleopRetargetingIK(
            self.model,
            left_hand_ik,
            right_hand_ik,
            body_active_joint_groups=[f"{arm_side}_arm", "waist_yaw_only"],
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
        self.solver.body_ik_solver.tasks["posture"].cost = 0.5

        upper_body = self.model.get_joint_group_indices("upper_body")
        local_index = {joint: index for index, joint in enumerate(upper_body)}
        self.hand_poses = {
            f"{arm_side}_hand_{name}_joint":
                poses if arm_side == "right" or name == "thumb_0"
                else tuple(-value for value in poses)
            for name, poses in HAND_POSES.items()
        }
        self.upper_joint_index = {
            name: local_index[self.model.dof_index(name)] for name in self.hand_poses
        }

        def group(name):
            return [local_index[joint] for joint in self.model.get_joint_group_indices(name)]

        self.waist_yaw = local_index[waist]
        self.arm = group(f"{arm_side}_arm")
        self.hand = group(f"{arm_side}_hand")
        self.inactive_arm = group(f"{self.other_side}_arm")
        self.inactive_hand = group(f"{self.other_side}_hand")
        self.inactive_arm_pose = np.array(
            [0.0, mirror * np.deg2rad(30.0), 0.0, np.pi / 2, 0.0, 0.0, 0.0]
        )
        self.joints = self.model.default_body_pose[upper_body].copy()
        self._apply_fixed_joints(self.joints)

    def _apply_fixed_joints(self, joints, grabbing=False):
        joints[self.inactive_arm] = self.inactive_arm_pose
        joints[self.inactive_hand] = 0.0
        for joint_name, poses in self.hand_poses.items():
            joints[self.upper_joint_index[joint_name]] = poses[grabbing]

    def ik(self, position, rotation, q, grabbing=False):
        target = np.eye(4)
        target[:3, 3] = np.array(position, dtype=float)
        target[:3, :3] = np.array(rotation, dtype=float)
        if self.arm_side == "left":
            target[:3, :3] = target[:3, :3] @ np.diag([1.0, -1.0, -1.0])

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
