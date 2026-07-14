import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.main.teleop.board_camera_geometry import board_cells_from_camera_depth
from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import instantiate_g1_hand_ik_solver
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
        self.cells_pelvis = board_cells_from_camera_depth()
        self.hover_offset = 0.16
        self.down_offset = 0.02
        self.cache_y = 0.13
        self.awaiting_move = False
        self.typed_move = ""
        self.palm_down_rot = np.diag([1.0, -1.0, -1.0]) @ R.from_euler("x", 90.0, degrees=True).as_matrix()
        self.robot.cache_forward_kinematics(self.robot.default_body_pose)
        self.target = self.robot.frame_placement(self.frame).homogeneous.copy()
        upper_body = list(self.robot.get_joint_group_indices("upper_body"))
        self.hand_slice = [upper_body.index(i) for i in self.robot.get_joint_group_indices("right_hand")]

    def refresh_board_from_robot_pose(self):
        state = self.state_subscriber.get_msg()
        if state is None or "floating_base_pose" not in state:
            self.cells_pelvis = board_cells_from_camera_depth()
            return
        pose = np.array(state["floating_base_pose"])
        rot = R.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        self.cells_pelvis = board_cells_from_camera_depth(pose[:3], rot)

    def board_cell_pelvis(self, key, offset=None):
        point = self.cells_pelvis[str(key)].copy()
        point[2] += self.hover_offset if offset is None else offset
        return point
    def cache_pelvis(self, offset=None):
        point = self.cells_pelvis["5"].copy() + np.array([0.0, self.cache_y, 0.0])
        point[2] += self.hover_offset if offset is None else offset
        return point
    def set_target(self, goal_pelvis):
        self.target[:3, 3] = goal_pelvis
        self.target[:3, :3] = self.palm_down_rot
    def set_goal_from_board_cell(self, key):
        self.refresh_board_from_robot_pose()
        goal_pelvis = self.board_cell_pelvis(key)
        self.set_target(goal_pelvis)
        print(f"goal cell {key}: pelvis {np.round(goal_pelvis, 3)}")
    def send(self, hand_q=None, duration=1.2):
        self.ik.set_goal(
            {"body_data": {self.frame: self.target}, "left_hand_data": None, "right_hand_data": None}
        )
        q = self.ik.get_action()
        q[self.hand_slice] = self.close_q if hand_q is None else hand_q
        self.publisher.publish({"target_upper_body_pose": q, "target_time": time.monotonic() + duration})
    def go(self, point_pelvis, hand_q=None, wait=1.4):
        self.set_target(point_pelvis)
        self.send(hand_q=hand_q, duration=wait)
        time.sleep(wait)

    def take_green_to_cell(self, key):
        print(f"moving green piece to {key}")
        self.refresh_board_from_robot_pose()
        self.go(self.board_cell_pelvis("5"), self.open_q)
        self.go(self.cache_pelvis(), self.open_q)
        self.go(self.cache_pelvis(self.down_offset), self.open_q)
        self.go(self.cache_pelvis(self.down_offset), self.close_q, wait=0.9)
        self.go(self.cache_pelvis(), self.close_q)
        self.go(self.board_cell_pelvis(key), self.close_q)
        self.go(self.board_cell_pelvis(key, self.down_offset), self.close_q)
        self.go(self.board_cell_pelvis(key, self.down_offset), self.open_q, wait=0.9)
        self.go(self.board_cell_pelvis(key), self.open_q)
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
