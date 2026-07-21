import numpy as np

from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from gr00t_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK


class Controller:
    def __init__(self, waist_yaw=0.0):
        self.model = instantiate_g1_robot_model(waist_location="upper_body")
        waist = self.model.dof_index("waist_yaw_joint")
        self.model.pinocchio_wrapper.q0[waist] = waist_yaw
        self.model.default_body_pose[waist] = waist_yaw
        self.model.initial_body_pose[waist] = waist_yaw
        self.site = "right_hand_palm_center"
        self.model.supplemental_info.hand_frame_names["right"] = self.site

        left_hand_ik, right_hand_ik = instantiate_g1_hand_ik_solver()
        self.solver = TeleopRetargetingIK(
            self.model,
            left_hand_ik,
            right_hand_ik,
            body_active_joint_groups=["right_arm"],
        )
        self.solver.body_ik_solver.num_step_per_frame = 30
        self.solver.body_ik_solver.update_weights(
            {self.site: {"position_cost": 80.0, "orientation_cost": 3.0}}
        )

        upper_body = list(self.model.get_joint_group_indices("upper_body"))
        self.waist_yaw = upper_body.index(waist)
        self.left_arm = [
            upper_body.index(i) for i in self.model.get_joint_group_indices("left_arm")
        ]
        self.left_hand = [
            upper_body.index(i) for i in self.model.get_joint_group_indices("left_hand")
        ]
        self.right_hand = [
            upper_body.index(i) for i in self.model.get_joint_group_indices("right_hand")
        ]
        self.joints = self.model.default_body_pose[
            self.model.get_joint_group_indices("upper_body")
        ].copy()
        self.hand = np.zeros(7)
        self.relax_left_arm(self.joints)

    def neutral(self):
        joints = self.joints.copy()
        joints[self.waist_yaw] = 0.0
        return joints

    def turn_left(self):
        return self.joints.copy()

    def relax_left_arm(self, joints):
        joints[self.left_arm] = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0])
        joints[self.left_hand] = np.zeros(7)

    def ik(self, position, rotation):
        target = np.eye(4)
        target[:3, 3] = np.array(position, dtype=float)
        target[:3, :3] = np.array(rotation, dtype=float)
        self.solver.set_goal(
            {"body_data": {self.site: target}, "left_hand_data": None, "right_hand_data": None}
        )
        self.joints = self.solver.get_action().copy()
        self.relax_left_arm(self.joints)
        self.joints[self.right_hand] = self.hand
        return self.joints

    def gripper(self, state):
        self.hand = (
            np.array([1.45, 1.65, 1.45, 1.65, 0.0, -0.9, -1.6])
            if state == "close"
            else np.zeros(7)
        )
        self.joints[self.right_hand] = self.hand
        self.relax_left_arm(self.joints)
        return self.joints

    def commands(self, plan):
        for step in plan:
            yield ("move", self.ik(step[1], step[2]), step[3]) if step[0] == "move" else step
