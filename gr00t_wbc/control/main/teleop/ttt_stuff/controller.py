import numpy as np

from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from gr00t_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK


class Controller:
    def __init__(self):
        self.model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
        self.site = "right_hand_palm_center"
        self.model.supplemental_info.hand_frame_names["right"] = self.site

        left_hand_ik, right_hand_ik = instantiate_g1_hand_ik_solver()
        self.solver = TeleopRetargetingIK(
            self.model,
            left_hand_ik,
            right_hand_ik,
            body_active_joint_groups=["right_arm", "waist"],
        )
        self.solver.body_ik_solver.num_step_per_frame = 30
        self.solver.body_ik_solver.update_weights(
            {self.site: {"position_cost": 80.0, "orientation_cost": 3.0}}
        )

        upper_body = list(self.model.get_joint_group_indices("upper_body"))
        self.left_arm = [
            upper_body.index(i) for i in self.model.get_joint_group_indices("left_arm")
        ]
        self.left_hand = [
            upper_body.index(i) for i in self.model.get_joint_group_indices("left_hand")
        ]
        self.right_hand = [
            upper_body.index(i) for i in self.model.get_joint_group_indices("right_hand")
        ]
        self.waist = [
            upper_body.index(i) for i in self.model.get_joint_group_indices("waist")
        ]
        self.waist_goal = np.zeros(3)
        self.joints = self.model.default_body_pose[
            self.model.get_joint_group_indices("upper_body")
        ].copy()
        self.hand = np.zeros(7)
        self.relax_left_arm(self.joints)
        self.hold_waist(self.joints)

        posture = self.solver.body_ik_solver.tasks["posture"]
        for i in self.solver.body.get_joint_group_indices("waist"):
            posture.weights[i] = 1000.0

    def relax_left_arm(self, joints):
        joints[self.left_arm] = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0])
        joints[self.left_hand] = np.zeros(7)

    def hold_waist(self, joints):
        joints[self.waist] = self.waist_goal

    def move_waist(self, key):
        if key == "waist_up":
            self.waist_goal[2] = min(self.waist_goal[2] + 0.1, 0.5)
        elif key == "waist_down":
            self.waist_goal[2] = max(self.waist_goal[2] - 0.1, -0.5)
        elif key == "waist_left":
            self.waist_goal[0] = min(self.waist_goal[0] + 0.1, 0.8)
        elif key == "waist_right":
            self.waist_goal[0] = max(self.waist_goal[0] - 0.1, -0.8)
        self.hold_waist(self.joints)
        print(f"Waist yaw {self.waist_goal[0]:.1f}, pitch {self.waist_goal[2]:.1f}", flush=True)
        return self.joints

    def set_solver_waist(self):
        configuration = self.solver.body_ik_solver.configuration
        for value, index in zip(
            self.waist_goal,
            self.solver.body.get_joint_group_indices("waist"),
        ):
            configuration.q[index] = value
        configuration.update()
        self.solver.body_ik_solver.tasks["posture"].set_target_from_configuration(configuration)

    def ik(self, position, rotation):
        self.set_solver_waist()
        target = np.eye(4)
        target[:3, 3] = np.array(position, dtype=float)
        target[:3, :3] = np.array(rotation, dtype=float)
        self.solver.set_goal(
            {"body_data": {self.site: target}, "left_hand_data": None, "right_hand_data": None}
        )
        self.joints = self.solver.get_action().copy()
        self.relax_left_arm(self.joints)
        self.hold_waist(self.joints)
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
        self.hold_waist(self.joints)
        return self.joints

    def commands(self, plan):
        for step in plan:
            yield ("move", self.ik(step[1], step[2]), step[3]) if step[0] == "move" else step
