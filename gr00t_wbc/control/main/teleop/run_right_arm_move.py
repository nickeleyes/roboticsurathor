import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import instantiate_g1_hand_ik_solver
from gr00t_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from gr00t_wbc.control.utils.keyboard_dispatcher import KeyboardDispatcher
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber


TABLE_XML = Path(__file__).resolve().parents[2] / "robot_model/model_data/tictactoe_table.xml"


class RightArmMove:
    def __init__(self):
        self.robot = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
        self.frame = "right_hand_palm_center"
        self.robot.supplemental_info.hand_frame_names["right"] = self.frame

        left_hand_ik, right_hand_ik = instantiate_g1_hand_ik_solver()
        self.ik = TeleopRetargetingIK(
            self.robot,
            left_hand_ik,
            right_hand_ik,
            body_active_joint_groups=["waist_pitch_only", "right_arm"],
        )
        self.ik.body_ik_solver.num_step_per_frame = 40
        self.ik.body_ik_solver.tasks = {
            name: task
            for name, task in self.ik.body_ik_solver.tasks.items()
            if name in {self.frame, "posture"}
        }
        self.ik.body_ik_solver.update_weights(
            {self.frame: {"position_cost": 120.0, "orientation_cost": 0.0}}
        )
        self._set_posture_weight("waist_pitch_joint", 0.05)

        self.publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
        self.state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)

        self.cells = self._load_cells()
        self.selected_cell = "5"
        self.height_offset = 0.0
        self.upper_body_indices = self.robot.get_joint_group_indices("upper_body")
        self.controlled_indices = self.robot.get_joint_group_indices(["waist_pitch_only", "right_arm"])

        self.robot.cache_forward_kinematics(self.robot.default_body_pose)
        self.target = self.robot.frame_placement(self.frame).homogeneous.copy()
        self.set_target("5")

    def _load_cells(self):
        root = ET.parse(TABLE_XML).getroot()
        table = root.find(".//body[@name='tictactoe_table']")
        table_pos = np.fromstring(table.attrib["pos"], sep=" ")
        cells = {}
        for geom in table.findall("geom"):
            name = geom.attrib.get("name", "")
            if name.startswith("cell_"):
                cells[name.removeprefix("cell_")] = table_pos + np.fromstring(
                    geom.attrib["pos"], sep=" "
                )
        return cells

    def _set_posture_weight(self, joint_name, weight):
        task = self.ik.body_ik_solver.tasks.get("posture")
        body = self.ik.body
        if task is not None and hasattr(task, "weights") and joint_name in body.joint_to_dof_index:
            task.weights[body.joint_to_dof_index[joint_name]] = weight

    def _state(self):
        state = self.state_subscriber.get_msg() or {}
        q = np.array(state.get("q", self.robot.default_body_pose), dtype=float)
        base = np.array(state.get("floating_base_pose", [0, 0, 0, 1, 0, 0, 0]), dtype=float)
        return q, base

    def _world_to_robot(self, point_world):
        _, base = self._state()
        base_pos = base[:3]
        base_rot = R.from_quat(base[[4, 5, 6, 3]]).as_matrix()
        return base_rot.T @ (point_world - base_pos)

    def set_target(self, cell):
        self.selected_cell = cell
        world = self.cells[cell].copy()
        world[2] += self.height_offset
        self.target[:3, 3] = self._world_to_robot(world)
        print(
            f"cell {cell} world {np.round(world, 3)} robot {np.round(self.target[:3, 3], 3)}",
            flush=True,
        )

    def send(self):
        current_q, _ = self._state()
        self.ik.set_goal(
            {"body_data": {self.frame: self.target}, "left_hand_data": None, "right_hand_data": None}
        )
        solved_upper = self.ik.get_action()

        command_q = current_q.copy()
        solved_full = self.robot.default_body_pose.copy()
        solved_full[self.upper_body_indices] = solved_upper
        command_q[self.controlled_indices] = solved_full[self.controlled_indices]

        self.publisher.publish(
            {
                "target_upper_body_pose": command_q[self.upper_body_indices],
                "target_time": time.monotonic() + 1.2,
            }
        )

    def handle_keyboard_button(self, key):
        key = key.lower()
        if key in self.cells:
            self.set_target(key)
        elif key == "i":
            self.height_offset += 0.01
            self.set_target(self.selected_cell)
        elif key == "k":
            self.height_offset -= 0.01
            self.set_target(self.selected_cell)
        elif key in ("space", " "):
            self.send()


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
