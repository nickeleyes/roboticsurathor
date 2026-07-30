import numpy as np
from pink.tasks import Task

from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from gr00t_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK

FIXED_GRIP_POSE = np.array([0.0, 0.0, 0.7, 0.8, -0.45, -0.45, -0.7])
LEFT_ARM_AT_SIDE = np.array(
    [0.0, np.deg2rad(30.0), 0.0, np.pi / 2, 0.0, 0.0, 0.0]
)


class PalmNormalTask(Task):
    def __init__(self, frame, cost):
        super().__init__(cost=cost)
        self.frame = frame
        self.target = np.array([0.0, 0.0, -1.0])

    def compute_error(self, configuration):
        rotation = configuration.get_transform_frame_to_world(self.frame).rotation
        normal_in_palm = rotation.T @ self.target
        return np.cross([0.0, 1.0, 0.0], normal_in_palm)

    def compute_jacobian(self, configuration):
        rotation = configuration.get_transform_frame_to_world(self.frame).rotation
        normal_in_palm = rotation.T @ self.target
        skew_y = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        skew_n = np.array(
            [
                [0, -normal_in_palm[2], normal_in_palm[1]],
                [normal_in_palm[2], 0, -normal_in_palm[0]],
                [-normal_in_palm[1], normal_in_palm[0], 0],
            ]
        )
        return skew_y @ skew_n @ configuration.get_frame_jacobian(self.frame)[3:]

    def __repr__(self):
        return f"PalmNormalTask({self.frame})"


class Controller:
    def __init__(self, waist_yaw):
        self.model = instantiate_g1_robot_model(waist_location="upper_body")
        waist = self.model.dof_index("waist_yaw_joint")
        self.model.pinocchio_wrapper.q0[waist] = waist_yaw
        self.model.default_body_pose[waist] = waist_yaw
        self.model.initial_body_pose[waist] = waist_yaw
        self.site = "right_hand_grip_center"
        self.model.supplemental_info.hand_frame_names["right"] = self.site

        left_hand_ik, right_hand_ik = instantiate_g1_hand_ik_solver()
        self.solver = TeleopRetargetingIK(
            self.model,
            left_hand_ik,
            right_hand_ik,
            body_active_joint_groups=["right_arm", "waist_yaw_only"],
        )
        self.solver.body_ik_solver.num_step_per_frame = 30
        self.solver.body_ik_solver.update_weights(
            {self.site: {"position_cost": 40.0, "orientation_cost": np.zeros(3)}}
        )
        self.normal_task = PalmNormalTask(self.site, 50.0)
        self.solver.body_ik_solver.tasks["palm_normal"] = self.normal_task

        upper_body = self.model.get_joint_group_indices("upper_body")
        local_index = {joint: index for index, joint in enumerate(upper_body)}

        def group(name):
            return [
                local_index[joint]
                for joint in self.model.get_joint_group_indices(name)
            ]

        self.waist_yaw = local_index[waist]
        self.left_arm = group("left_arm")
        self.right_arm = group("right_arm")
        self.left_hand = group("left_hand")
        self.right_hand = group("right_hand")
        self.joints = self.model.default_body_pose[upper_body].copy()
        # index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2.
        # The index finger stays fully extended; the middle finger forms the pinch.
        self._apply_fixed_joints(self.joints)

    def _apply_fixed_joints(self, joints):
        joints[self.left_arm] = LEFT_ARM_AT_SIDE
        joints[self.left_hand] = 0.0
        joints[self.right_hand] = FIXED_GRIP_POSE

    def ik(self, position, rotation):
        self.normal_task.target = np.asarray(rotation, dtype=float)[:, 1]
        target = np.eye(4)
        target[:3, 3] = np.array(position, dtype=float)
        target[:3, :3] = np.array(rotation, dtype=float)
        self.solver.set_goal(
            {"body_data": {self.site: target}, "left_hand_data": None, "right_hand_data": None}
        )
        self.joints = self.solver.get_action().copy()
        self._apply_fixed_joints(self.joints)
        return self.joints
