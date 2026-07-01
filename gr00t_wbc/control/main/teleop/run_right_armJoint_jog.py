import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from gr00t_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from gr00t_wbc.control.utils.keyboard_dispatcher import KeyboardDispatcher
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber


class RightArmIKJog:
    def __init__(self):
        self.robot = instantiate_g1_robot_model()
        self.robot.supplemental_info.hand_frame_names["right"] = "right_hand_palm_link"
        left_hand_ik, right_hand_ik = instantiate_g1_hand_ik_solver()
        self.ik = TeleopRetargetingIK(
            self.robot, left_hand_ik, right_hand_ik, body_active_joint_groups=["upper_body"]
        )
        self.publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
        self.state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
        self.frame = self.robot.supplemental_info.hand_frame_names["right"]
        self.last_base_pos = np.zeros(3)
        self.last_base_rot = np.eye(3)
        self.have_base_pose = False
        self.close_q = np.array([1.45, 1.65, 1.45, 1.65, 0.0, -0.9, -1.6])
        self.hover = np.array([0.0, 0.0, 0.30])
        table = np.array([1.1, 0.0, 0.6])
        self.board = table + np.array([
            [0.108, 0.1512, 0.048],
            [0.108, -0.1512, 0.048],
            [-0.108, 0.1512, 0.048],
            [-0.108, -0.1512, 0.048],
        ])
        self.robot.cache_forward_kinematics(self.robot.default_body_pose)
        self.target = self.robot.frame_placement(self.frame).homogeneous.copy()
        upper_body = list(self.robot.get_joint_group_indices("upper_body"))
        self.hand_slice = [upper_body.index(i) for i in self.robot.get_joint_group_indices("right_hand")]
        self.point_hand()

    def base_pose(self):
        state = self.state_subscriber.get_msg()
        if state is not None and "floating_base_pose" in state:
            q = np.array(state["floating_base_pose"][3:7])
            self.last_base_pos = np.array(state["floating_base_pose"][:3])
            self.last_base_rot = R.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            self.have_base_pose = True
        elif not self.have_base_pose:
            print("no base pose yet; assuming world origin")
            self.have_base_pose = True
        return self.last_base_pos, self.last_base_rot

    def point_hand(self):
        normal = np.array([0.0, 1.0, 0.0])
        _, base_rot_world = self.base_pose()
        if base_rot_world is not None:
            normal = base_rot_world.T @ normal

        x_axis = self.target[:3, 0] - normal * np.dot(self.target[:3, 0], normal)
        x_axis /= np.linalg.norm(x_axis)
        self.target[:3, :3] = np.column_stack([x_axis, np.cross(normal, x_axis), normal])

    def world_to_robot(self, point_world):
        base_pos, base_rot_world = self.base_pose()
        if base_pos is None:
            return None
        return base_rot_world.T @ (point_world - base_pos)

    def board_rotation_robot(self):
        _, base_rot_world = self.base_pose()
        if base_rot_world is None:
            return None
        left_to_right = self.board[1] - self.board[0]
        front_to_back = self.board[2] - self.board[0]
        x_axis = left_to_right / np.linalg.norm(left_to_right)
        y_axis = front_to_back / np.linalg.norm(front_to_back)
        z_axis = np.cross(y_axis, x_axis)
        z_axis /= np.linalg.norm(z_axis)
        return base_rot_world.T @ np.column_stack([x_axis, y_axis, z_axis])

    def board_cell_world(self, key):
        row, col = divmod(int(key) - 1, 3)
        u, v = (col + 0.5) / 3.0, (row + 0.5) / 3.0
        front = (1 - u) * self.board[0] + u * self.board[1]
        back = (1 - u) * self.board[2] + u * self.board[3]
        return ((1 - v) * front + v * back) + self.hover

    def send(self):
        self.ik.set_goal(
            {"body_data": {self.frame: self.target}, "left_hand_data": None, "right_hand_data": None}
        )
        q = self.ik.get_action()
        q[self.hand_slice] = self.close_q
        self.publisher.publish({"target_upper_body_pose": q, "target_time": time.monotonic() + 0.5})

    def handle_keyboard_button(self, key):
        moves = {
            "j": (0, -0.02),
            "k": (0, 0.02),
            "l": (1, -0.02),
            ";": (1, 0.02),
            ",": (2, -0.02),
            ".": (2, 0.02),
        }

        if key in moves:
            axis, step = moves[key]
            self.target[axis, 3] += step
        elif key in "123456789":
            point_robot = self.world_to_robot(self.board_cell_world(key))
            board_rot_robot = self.board_rotation_robot()
            if point_robot is not None and board_rot_robot is not None:
                self.target[:3, 3] = point_robot
                self.target[:3, :3] = board_rot_robot
        elif key == "i":
            self.send()


def main():
    ros = ROSManager(node_name="RightArmIKJog")
    keyboard = KeyboardDispatcher()
    keyboard.register(RightArmIKJog())
    keyboard.start()
    try:
        while ros.ok():
            time.sleep(0.1)
    except ros.exceptions():
        pass
    keyboard.stop()
    ros.shutdown()

if __name__ == "__main__":
    main()
