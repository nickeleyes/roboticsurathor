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


class RightArmMove:
    def __init__(self):
        self.robot = instantiate_g1_robot_model()
        self.frame = "right_hand_palm_center"
        self.robot.supplemental_info.hand_frame_names["right"] = self.frame
        left_hand_ik, right_hand_ik = instantiate_g1_hand_ik_solver()
        self.ik = TeleopRetargetingIK(
            self.robot, left_hand_ik, right_hand_ik, body_active_joint_groups=["upper_body"]
        )
        self.ik.body_ik_solver.num_step_per_frame = 30
        self.ik.body_ik_solver.update_weights(
            {self.frame: {"position_cost": 80.0, "orientation_cost": 3.0}}
        )
        self.publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
        self.state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
        self.close_q = np.array([1.45, 1.65, 1.45, 1.65, 0.0, -0.9, -1.6])
        self.open_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.base_pos = np.zeros(3)
        self.base_rot = np.eye(3)
        self.table_world = np.array([0.65, -0.25, 0.55])
        self.cell_step = 0.05
        self.hover_height = 0.24
        self.down_height = 0.08
        self.cache_y = 0.13
        self.awaiting_move = False
        self.typed_move = ""
        self.palm_down_rot = np.diag([1.0, -1.0, -1.0]) @ R.from_euler(
            "x", 90.0, degrees=True
        ).as_matrix()

        self.robot.cache_forward_kinematics(self.robot.default_body_pose)
        self.target = self.robot.frame_placement(self.frame).homogeneous.copy()
        upper_body = list(self.robot.get_joint_group_indices("upper_body"))
        self.hand_slice = [upper_body.index(i) for i in self.robot.get_joint_group_indices("right_hand")]

    def base_pose(self):
        state = self.state_subscriber.get_msg()
        if state is not None and "floating_base_pose" in state:
            q_wxyz = np.array(state["floating_base_pose"][3:7])
            self.base_pos = np.array(state["floating_base_pose"][:3])
            self.base_rot = R.from_quat(q_wxyz[[1, 2, 3, 0]]).as_matrix()
        return self.base_pos, self.base_rot

    def board_cell_world(self, key, height=None):
        row, col = divmod(int(key) - 1, 3)
        cell_local = np.array([
            self.cell_step * (1 - row),
            self.cell_step * (1 - col),
            self.hover_height if height is None else height,
        ])
        return self.table_world + cell_local

    def cache_world(self, height=None):
        return self.table_world + np.array([
            0.0,
            self.cache_y,
            self.hover_height if height is None else height,
        ])

    def set_target(self, goal_world):
        base_pos, base_rot = self.base_pose()
        self.target[:3, 3] = base_rot.T @ (goal_world - base_pos)
        self.target[:3, :3] = self.palm_down_rot

    def set_goal_from_board_cell(self, key):
        goal_world = self.board_cell_world(key)
        self.set_target(goal_world)
        print(f"goal cell {key}: world {np.round(goal_world, 3)} robot {np.round(self.target[:3, 3], 3)}")

    def send(self, hand_q=None, duration=1.2):
        self.ik.set_goal(
            {"body_data": {self.frame: self.target}, "left_hand_data": None, "right_hand_data": None}
        )
        q = self.ik.get_action()
        q[self.hand_slice] = self.close_q if hand_q is None else hand_q
        self.publisher.publish({"target_upper_body_pose": q, "target_time": time.monotonic() + duration})

    def go(self, point_world, hand_q=None, wait=1.4):
        self.set_target(point_world)
        self.send(hand_q=hand_q, duration=wait)
        time.sleep(wait)

    def take_green_to_cell(self, key):
        print(f"moving green piece to {key}")
        self.go(self.board_cell_world("5"), self.open_q)
        self.go(self.cache_world(), self.open_q)
        self.go(self.cache_world(self.down_height), self.open_q)
        self.go(self.cache_world(self.down_height), self.close_q, wait=0.9)
        self.go(self.cache_world(), self.close_q)
        self.go(self.board_cell_world(key), self.close_q)
        self.go(self.board_cell_world(key, self.down_height), self.close_q)
        self.go(self.board_cell_world(key, self.down_height), self.open_q, wait=0.9)
        self.go(self.board_cell_world(key), self.open_q)
        print("done")

    def handle_keyboard_button(self, key):
        key = key.lower()
        if self.awaiting_move:
            self.typed_move += key
            if self.typed_move[-1] in "123456789":
                piece, cell = self.typed_move[0], self.typed_move[-1]
                self.awaiting_move = False
                self.typed_move = ""
                if piece == "g":
                    self.take_green_to_cell(cell)
            return

        if key == "t":
            self.awaiting_move = True
            self.typed_move = ""
            print("type move, like g6", flush=True)
        elif key in "123456789":
            self.set_goal_from_board_cell(key)
        elif key in ("space", " "):
            self.send()
            print("sent")


def main():
    ros = ROSManager(node_name="RightArmMove")
    keyboard = KeyboardDispatcher()
    keyboard.register(RightArmMove())
    keyboard.start()
    try:
        while ros.ok():
            time.sleep(0.1)
    except ros.exceptions():
        pass
    finally:
        keyboard.stop()
        ros.shutdown()


if __name__ == "__main__":
    main()
