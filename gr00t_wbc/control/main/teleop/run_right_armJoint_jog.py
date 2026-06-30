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
        left_hand_ik, right_hand_ik = instantiate_g1_hand_ik_solver()
        self.ik = TeleopRetargetingIK(
            self.robot, left_hand_ik, right_hand_ik, body_active_joint_groups=["upper_body"]
        )
        self.publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
        self.state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
        self.frame = self.robot.supplemental_info.hand_frame_names["right"]
        self.closed = False
        self.open_q = np.zeros(7)
        self.close_q = np.array([1.45, 1.65, 0.0, 0.0, 0.0, -0.9, -1.6]) * (4.5 / 8)
        self.close_q[4] += 0.45

        self.robot.cache_forward_kinematics(self.robot.default_body_pose)
        self.target = self.robot.frame_placement(self.frame).homogeneous.copy()
        upper_body = list(self.robot.get_joint_group_indices("upper_body"))
        self.hand_slice = [
            upper_body.index(i) for i in self.robot.get_joint_group_indices("right_hand")
        ]
        self.point_hand()
        print(f"target xyz: {np.round(self.target[:3, 3], 3)}")

    def point_hand(self):
        normal = np.array([0.0, 1.0, 0.0])
        state = self.state_subscriber.get_msg()
        if state is not None and "floating_base_pose" in state:
            q = np.array(state["floating_base_pose"][3:7])
            normal = R.from_quat(q[[1, 2, 3, 0]]).as_matrix().T @ normal
        x_axis = self.target[:3, 0] - normal * np.dot(self.target[:3, 0], normal)
        x_axis /= np.linalg.norm(x_axis)
        self.target[:3, :3] = np.column_stack([x_axis, np.cross(normal, x_axis), normal])

    def handle_keyboard_button(self, key):
        moves = {"j": (0, -0.02), "k": (0, 0.02), "l": (1, -0.02), ";": (1, 0.02), ",": (2, -0.02), ".": (2, 0.02)}
        if key in moves:
            axis, step = moves[key]
            self.target[axis, 3] += step
        elif key == "q":
            self.closed = not self.closed
            print("right gripper:", "close" if self.closed else "open")
        elif key == "i":
            self.point_hand()
            self.ik.set_goal({"body_data": {self.frame: self.target}, "left_hand_data": None, "right_hand_data": None})
            q = self.ik.get_action()
            q[self.hand_slice] = self.close_q if self.closed else self.open_q
            self.publisher.publish({"target_upper_body_pose": q, "target_time": time.monotonic() + 0.5})
        else:
            return
        print(f"target xyz: {np.round(self.target[:3, 3], 3)}")


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
    finally:
        keyboard.stop()
        ros.shutdown()


if __name__ == "__main__":
    main()
