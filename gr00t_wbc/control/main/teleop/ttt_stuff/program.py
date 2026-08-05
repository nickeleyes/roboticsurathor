import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.main.teleop.ttt_stuff.controller import Controller
from gr00t_wbc.control.main.teleop.ttt_stuff.engine import move_code
from gr00t_wbc.control.main.teleop.ttt_stuff.localization import localize
from gr00t_wbc.control.main.teleop.ttt_stuff.pelvis_tracking import PelvisTracker
from gr00t_wbc.control.main.teleop.ttt_stuff.planner import BoardPlanner
from gr00t_wbc.control.main.teleop.ttt_stuff.ros2capture import capture
from gr00t_wbc.control.main.teleop.ttt_stuff.vision import detect
from gr00t_wbc.control.utils.ros_utils import ROSMsgPublisher, ROSMsgSubscriber

POSE_DURATION = 5.0
KEEPALIVE_PERIOD = 0.4
CAPTURE_DIR = Path(__file__).resolve().parents[5] / "camera_captures"
RIGHT_ARM_POSES = (
    ("shoulder roll 30, elbow 90", [0.0, np.deg2rad(-30), 0.0, np.pi / 2, 0.0, 0.0, 0.0]),
    ("shoulder roll 90, elbow 90", [0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0, 0.0]),
    ("shoulder roll 90, elbow 0", [0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0]),
)


class TTTProgram:
    def __init__(self, config):
        self.config = config
        self.controller = Controller()
        self.planner = BoardPlanner()
        self.pelvis = PelvisTracker() if config.env_type == "real" else None
        self.publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
        self.state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
        self.initialized = threading.Event()
        self.continue_event = threading.Event()
        self.move_thread = None
        self.ik_period = 1.0 / config.control_frequency
        self.upper_indices = self.controller.model.get_joint_group_indices("upper_body")
        self.controlled = np.array([self.controller.waist_yaw, *self.controller.right_arm])

    def _publish(self, joints=None, duration=None, marker=None):
        goal = {
            "navigate_cmd": np.zeros(3, dtype=np.float32),
            "preserve_upper_body_waist_yaw": True,
        }
        if joints is not None:
            goal["target_upper_body_pose"] = joints
            goal["target_time"] = time.monotonic() + duration
        if marker is not None:
            goal["ttt_ik_target"] = marker.tolist()
        self.publisher.publish(goal)

    def _hold(self, duration):
        deadline = time.monotonic() + duration
        while (remaining := deadline - time.monotonic()) > 0:
            time.sleep(min(KEEPALIVE_PERIOD, remaining))
            if time.monotonic() < deadline:
                self._publish()

    def _arm_pose(self, right_arm):
        joints = self.controller.joints.copy()
        joints[self.controller.waist_yaw] = 0.0
        joints[self.controller.right_arm] = right_arm
        return joints

    def initialize(self):
        def run():
            for index, (label, pose) in enumerate(RIGHT_ARM_POSES, 1):
                target = self._arm_pose(pose)
                print(f"Initialization {index}/3: {label}", flush=True)
                self._publish(target, POSE_DURATION)
                while True:
                    time.sleep(KEEPALIVE_PERIOD)
                    state = self._state(KEEPALIVE_PERIOD)
                    if state is not None:
                        q = np.asarray(state["q"])
                        if q.shape == self.controller.model.default_body_pose.shape:
                            current = q[self.upper_indices]
                            error = np.max(np.abs(current[self.controlled] - target[self.controlled]))
                            if error <= np.deg2rad(2.0):
                                break
                    self._publish()
            self.initialized.set()
            print("Arm initialization complete; press p to capture the board", flush=True)

        threading.Thread(target=run, daemon=True).start()

    def continue_move(self):
        self.continue_event.set()

    def start_move(self):
        if not self.initialized.is_set():
            print("Arm initialization is still running", flush=True)
            return
        if self.move_thread is not None and self.move_thread.is_alive():
            print("A tic-tac-toe move is already running", flush=True)
            return
        if self.config.env_type == "sim":
            self.publisher.publish({"ttt_lock_feet": True})
        self.move_thread = threading.Thread(target=self._run_move, daemon=True)
        self.move_thread.start()

    def _state(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.state_subscriber.get_msg()
            if state is not None and state.get("q") is not None:
                return state
            time.sleep(0.001)
        return None

    def _capture_board(self):
        if self.config.offline:
            frame = cv2.imread(str(CAPTURE_DIR / "color_rgb.png"))
            if frame is None:
                raise FileNotFoundError(CAPTURE_DIR / "color_rgb.png")
            radial = np.load(CAPTURE_DIR / "radial_distance_m.npy")
            intrinsics = json.loads((CAPTURE_DIR / "camera_intrinsics.json").read_text())
            print(f"Using saved camera capture: {CAPTURE_DIR}", flush=True)
        else:
            frame, radial, intrinsics = capture()
        state = self._state()
        if state is None:
            raise RuntimeError("Robot state unavailable at camera capture")
        board, corners = localize(detect(frame), radial, intrinsics)
        return board, corners, move_code(board), state

    def _capture_context(self, corners, state):
        context = {"corners": corners}
        if self.config.env_type == "real":
            self.pelvis.capture(state["q"])
        else:
            base = np.asarray(state.get("floating_base_pose"))
            if base.shape != (7,):
                raise RuntimeError("Simulation pelvis pose unavailable")
            context["rotation"] = Rotation.from_quat(base[[4, 5, 6, 3]]).as_matrix()
            context["position"] = base[:3]
        print(f"Board corners in captured pelvis frame:\n{np.round(corners, 4)}", flush=True)
        return context

    def _fixed_target(self, context, name, height):
        position, rotation = self.planner.target(context["corners"], name, height)
        if self.config.env_type == "sim":
            position = context["rotation"] @ position + context["position"]
            rotation = context["rotation"] @ rotation
        return name, height, position, rotation

    def _current_target(self, target, state):
        position, rotation = target[2:]
        if self.config.env_type == "real":
            delta_rotation, delta_position = self.pelvis.delta(state["q"])
            return delta_rotation @ position + delta_position, delta_rotation @ rotation
        base = np.asarray(state.get("floating_base_pose"))
        if base.shape != (7,):
            raise RuntimeError("Simulation pelvis pose unavailable")
        pelvis_rotation = Rotation.from_quat(base[[4, 5, 6, 3]]).as_matrix()
        return pelvis_rotation.T @ (position - base[:3]), pelvis_rotation.T @ rotation

    def _run_waypoint(self, target):
        marker = target[2] if self.config.env_type == "sim" else None
        last_log = 0.0
        while not self.continue_event.is_set():
            started = time.monotonic()
            state = self._state(self.ik_period)
            if state is None:
                self._publish()
            else:
                position, rotation = self._current_target(target, state)
                full_q = np.asarray(state["q"])
                log = started - last_log >= 0.5
                if log:
                    self.controller.model.cache_forward_kinematics(full_q, auto_clip=False)
                    grip = self.controller.model.frame_placement(self.controller.site)
                    error = np.linalg.norm(grip.translation - position)
                    normal_error = np.arccos(np.clip(grip.rotation[:, 1] @ rotation[:, 1], -1.0, 1.0))
                    joint_speed = np.max(np.abs(np.asarray(state["dq"])[self.upper_indices][self.controlled]))
                solved = self.controller.ik(position, rotation, full_q)
                self._publish(solved, self.ik_period, marker)
                marker = None
                if log:
                    print(
                        f"Live IK {target[0]} {target[1]}: error={100 * error:.2f} cm, "
                        f"normal={np.rad2deg(normal_error):.1f} deg, "
                        f"motion={np.rad2deg(joint_speed):.2f} deg/s",
                        flush=True,
                    )
                    last_log = started
            remaining = self.ik_period - (time.monotonic() - started)
            if remaining > 0:
                self.continue_event.wait(remaining)

    def _run_move(self):
        print("Capturing the tic-tac-toe board", flush=True)
        try:
            board, corners, move, capture_state = self._capture_board()
            context = self._capture_context(corners, capture_state)
        except Exception as error:
            print(f"Board capture cancelled: {error}", flush=True)
            return
        if not move:
            print("No legal move available", flush=True)
            return

        steps = self.planner.plan_steps(board, move)
        print(f"Board {board}: playing {move}", flush=True)
        print(
            f"Live IK runs at {self.config.control_frequency} Hz with a 10 deg/s joint-speed limit",
            flush=True,
        )
        self.continue_event.clear()
        print(f"Waypoint 1/{len(steps)} is ready. Press SPACE to start.", flush=True)
        while not self.continue_event.wait(KEEPALIVE_PERIOD):
            self._publish()

        for index, (name, height) in enumerate(steps, 1):
            self.continue_event.clear()
            action = "finish" if index == len(steps) else "advance"
            print(
                f"Waypoint {index}/{len(steps)}: {name} {height}; press SPACE to {action} when satisfied.",
                flush=True,
            )
            self._run_waypoint(self._fixed_target(context, name, height))

        self._publish(self._arm_pose(RIGHT_ARM_POSES[-1][1]), POSE_DURATION)
        self._hold(POSE_DURATION)
        print("Tic-tac-toe move complete", flush=True)
