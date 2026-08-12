import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.main.teleop.ttt_stuff.controller import Controller
from gr00t_wbc.control.main.teleop.ttt_stuff.engine import move_code
from gr00t_wbc.control.main.teleop.ttt_stuff.pelvis_tracking import PelvisTracker
from gr00t_wbc.control.main.teleop.ttt_stuff.planner import BoardPlanner
from gr00t_wbc.control.utils.ros_utils import ROSMsgPublisher, ROSMsgSubscriber

POSE_DURATION = 5.0
KEEPALIVE_PERIOD = 0.4
WAIST_RELEASE_SETTLE_DURATION = 2.0
AUTO_SETTLE_DURATION = 0.5
AUTO_POSITION_TOLERANCE = 0.01
AUTO_ORIENTATION_TOLERANCE = np.deg2rad(8.0)
AUTO_SPEED_TOLERANCE = np.deg2rad(2.0)
OFFLINE_BOARD = "000000000"
# Pelvis-frame cell centers: p1 green-left, p3 white-right,
# p7 green-right, and p9 white-left.
OFFLINE_CORNERS = np.array([
    [0.34, 0.09, -0.03],
    [0.52, 0.09, -0.03],
    [0.34, -0.09, -0.03],
    [0.52, -0.09, -0.03],
])
ARM_POSES = (
    ("shoulder roll 30, elbow 90", [0.0, np.deg2rad(-30), 0.0, np.pi / 2, 0.0, 0.0, 0.0]),
    ("shoulder roll 90, elbow 90", [0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0, 0.0]),
    ("shoulder roll 90, elbow 0", [0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0]),
)


class TTTProgram:
    def __init__(self, config):
        self.config = config
        self.controller = Controller(config.arm_side)
        mirror = 1 if config.arm_side == "right" else -1
        self.arm_poses = tuple(
            (label, np.asarray(pose) * [1, mirror, 1, 1, 1, 1, 1])
            for label, pose in ARM_POSES
        )
        self.planner = BoardPlanner()
        self.pelvis = PelvisTracker() if config.env_type == "real" else None
        self.publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
        self.state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
        self.initialized = threading.Event()
        self.continue_event = threading.Event()
        self.automatic = threading.Event()
        self.shutdown_event = threading.Event()
        self._continue_lock = threading.Lock()
        self._ownership_lock = threading.Lock()
        self._waiting_for_continue = False
        self._waist_owned = False
        self.initialize_thread = None
        self.move_thread = None
        self.ik_period = 1.0 / config.control_frequency
        self.upper_indices = self.controller.model.get_joint_group_indices("upper_body")
        self.controlled = np.array([self.controller.waist_yaw, *self.controller.arm])

    def _publish(self, joints=None, duration=None, marker=None):
        goal = {
            "preserve_upper_body_waist_yaw": True,
        }
        if joints is not None:
            goal["target_upper_body_pose"] = joints
            goal["target_time"] = time.monotonic() + duration
        if marker is not None:
            goal["ttt_ik_target"] = marker.tolist()
        with self._ownership_lock:
            if self.shutdown_event.is_set():
                return
            self.publisher.publish(goal)
            self._waist_owned = True

    def _set_waist_ownership(self, enabled, lock_feet=False):
        """Select which policy owns waist yaw without changing an arm target."""
        goal = {"preserve_upper_body_waist_yaw": bool(enabled)}
        if lock_feet:
            goal["ttt_lock_feet"] = True
        with self._ownership_lock:
            if enabled and self.shutdown_event.is_set():
                return False
            self.publisher.publish(goal)
            self._waist_owned = bool(enabled)
            return True

    def _hold(self, duration):
        deadline = time.monotonic() + duration
        while not self.shutdown_event.is_set() and (remaining := deadline - time.monotonic()) > 0:
            self.shutdown_event.wait(min(KEEPALIVE_PERIOD, remaining))
            if not self.shutdown_event.is_set() and time.monotonic() < deadline:
                self._publish()

    def _arm_pose(self, arm):
        joints = self.controller.joints.copy()
        joints[self.controller.waist_yaw] = 0.0
        joints[self.controller.arm] = arm
        return joints

    def initialize(self):
        def run():
            try:
                for index, (label, pose) in enumerate(self.arm_poses, 1):
                    target = self._arm_pose(pose)
                    print(f"Initialization {index}/3: {label}", flush=True)
                    self._publish(target, POSE_DURATION)
                    while not self.shutdown_event.is_set():
                        self.shutdown_event.wait(KEEPALIVE_PERIOD)
                        state = self._state(KEEPALIVE_PERIOD)
                        if state is not None:
                            q = np.asarray(state["q"])
                            if q.shape == self.controller.model.default_body_pose.shape:
                                current = q[self.upper_indices]
                                error = np.max(
                                    np.abs(current[self.controlled] - target[self.controlled])
                                )
                                if error <= np.deg2rad(2.0):
                                    break
                        if not self.shutdown_event.is_set():
                            self._publish()
                    if self.shutdown_event.is_set():
                        return

                # Give waist yaw back to the lower-body controller before board capture.
                # Any resulting stance correction therefore finishes before p is enabled.
                self._set_waist_ownership(False)
                if self.shutdown_event.wait(WAIST_RELEASE_SETTLE_DURATION):
                    return
                self.initialized.set()
                print(
                    "Arm initialization complete and waist released; "
                    "press p to capture the board",
                    flush=True,
                )
            except Exception as error:
                print(f"Arm initialization cancelled: {error}", flush=True)
                if self._waist_owned:
                    self._set_waist_ownership(False)

        self.initialize_thread = threading.Thread(target=run, daemon=True)
        self.initialize_thread.start()

    def continue_move(self):
        with self._continue_lock:
            if not self._waiting_for_continue:
                print("SPACE ignored: no tic-tac-toe waypoint is ready", flush=True)
                return
            self.continue_event.set()

    def toggle_automatic(self):
        if self.automatic.is_set():
            self.automatic.clear()
            print("Automatic waypoints OFF; press SPACE to advance", flush=True)
        else:
            self.automatic.set()
            print("Automatic waypoints ON; advancing after each pose settles", flush=True)

    def _arm_continue_gate(self):
        with self._continue_lock:
            self.continue_event.clear()
            self._waiting_for_continue = True

    def _disarm_continue_gate(self):
        with self._continue_lock:
            self._waiting_for_continue = False
            self.continue_event.clear()

    def _wait_for_continue(self):
        self._arm_continue_gate()
        try:
            while not self.shutdown_event.is_set():
                if self.automatic.is_set() or self.continue_event.wait(0.1):
                    return True
            return False
        finally:
            self._disarm_continue_gate()

    def start_move(self):
        if not self.initialized.is_set():
            print("Arm initialization is still running", flush=True)
            return
        if self.move_thread is not None and self.move_thread.is_alive():
            print("A tic-tac-toe move is already running", flush=True)
            return
        self.move_thread = threading.Thread(target=self._run_move, daemon=True)
        self.move_thread.start()

    def wait_for_state(self, timeout=2.0):
        """Wait for a well-formed observation from the dedicated WBC process."""
        state = self._state(timeout)
        if state is None:
            return False
        return np.asarray(state["q"]).shape == self.controller.model.default_body_pose.shape

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
            state = self._state()
            board, corners = OFFLINE_BOARD, OFFLINE_CORNERS.copy()
            print("Using fixed offline board geometry (vision disabled)", flush=True)
        else:
            from gr00t_wbc.control.main.teleop.ttt_stuff.localization import localize
            from gr00t_wbc.control.main.teleop.ttt_stuff.ros2capture import capture
            from gr00t_wbc.control.main.teleop.ttt_stuff.vision import detect

            frame, radial, intrinsics, state = capture(self._state)
            camera_to_pelvis = None
            if self.config.env_type == "real":
                camera_to_pelvis = self.pelvis.camera_to_pelvis(state["q"])
                waist_deg = np.rad2deg(np.asarray(state["q"])[self.pelvis.waist])
                print(f"Camera capture waist yaw/roll/pitch: {np.round(waist_deg, 2)} deg", flush=True)
            board, corners = localize(
                detect(frame), radial, intrinsics, camera_to_pelvis
            )
        if state is None:
            raise RuntimeError("Robot state unavailable at camera capture")
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

    def _run_waypoint(self, target, grabbing):
        marker = target[2] if self.config.env_type == "sim" else None
        last_log = 0.0
        settled_since = None
        self._arm_continue_gate()
        try:
            while not self.continue_event.is_set() and not self.shutdown_event.is_set():
                started = time.monotonic()
                state = self._state(self.ik_period)
                if state is None:
                    self._publish()
                else:
                    position, rotation = self._current_target(target, state)
                    full_q = np.asarray(state["q"])
                    log = started - last_log >= 0.5
                    measure = log or self.automatic.is_set()
                    if measure:
                        self.controller.model.cache_forward_kinematics(full_q, auto_clip=False)
                        grip = self.controller.model.frame_placement(self.controller.site)
                        error = np.linalg.norm(grip.translation - position)
                        normal_error = np.arccos(
                            np.clip(grip.rotation[:, 1] @ rotation[:, 1], -1.0, 1.0)
                        )
                        joint_speed = np.max(
                            np.abs(
                                np.asarray(state["dq"])[self.upper_indices][self.controlled]
                            )
                        )
                    solved = self.controller.ik(position, rotation, full_q, grabbing)
                    self._publish(solved, 0.10, marker)
                    marker = None
                    settled = measure and (
                        error <= AUTO_POSITION_TOLERANCE
                        and normal_error <= AUTO_ORIENTATION_TOLERANCE
                        and joint_speed <= AUTO_SPEED_TOLERANCE
                    )
                    if self.automatic.is_set() and settled:
                        settled_since = settled_since or started
                        if started - settled_since >= AUTO_SETTLE_DURATION:
                            return
                    else:
                        settled_since = None
                    if log:
                        print(
                            f"Live IK {target[0]} {target[1]}: "
                            f"error={100 * error:.2f} cm, "
                            f"normal={np.rad2deg(normal_error):.1f} deg, "
                            f"motion={np.rad2deg(joint_speed):.2f} deg/s",
                            flush=True,
                        )
                        last_log = started
                remaining = self.ik_period - (time.monotonic() - started)
                if remaining > 0:
                    self.shutdown_event.wait(remaining)
        finally:
            self._disarm_continue_gate()

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
        if self.shutdown_event.is_set():
            return

        steps = self.planner.plan_steps(board, move)
        print(f"Board {board}: playing {move}", flush=True)
        print(
            "The first 19 waypoints check white depots 1-5, board positions "
            "1-9, and green depots 1-5 at 2 cm before piece handling",
            flush=True,
        )
        print(
            f"Live IK runs at {self.config.control_frequency} Hz with a 10 deg/s joint-speed limit",
            flush=True,
        )
        print(f"Waypoint 1/{len(steps)} is ready. Press SPACE to start.", flush=True)
        if not self._wait_for_continue():
            return

        waist_acquired = False
        completed = False
        try:
            # Capture and inference above intentionally publish no control goals. Waist
            # ownership and the sim foot lock begin only when SPACE starts motion.
            if self.config.env_type == "sim":
                waist_acquired = self._set_waist_ownership(True, lock_feet=True)
            else:
                waist_acquired = self._set_waist_ownership(True)
            if not waist_acquired:
                return

            for index, (name, height, grabbing) in enumerate(steps, 1):
                action = "finish" if index == len(steps) else "advance"
                hand = "grab" if grabbing else "open"
                print(
                    f"Waypoint {index}/{len(steps)}: {name} {height}, hand {hand}; "
                    f"press SPACE to {action} when satisfied.",
                    flush=True,
                )
                self._run_waypoint(self._fixed_target(context, name, height), grabbing)
                if self.shutdown_event.is_set():
                    break

            if not self.shutdown_event.is_set():
                self._publish(self._arm_pose(self.arm_poses[-1][1]), POSE_DURATION)
                self._hold(POSE_DURATION)
                completed = not self.shutdown_event.is_set()
        except Exception as error:
            print(f"Tic-tac-toe move cancelled: {error}", flush=True)
        finally:
            if waist_acquired and self._waist_owned:
                self._set_waist_ownership(False)

        if completed:
            print("Tic-tac-toe move complete; waist released", flush=True)

    def shutdown(self):
        """Stop task threads and ensure waist ownership is not left latched."""
        self.shutdown_event.set()
        self.continue_event.set()
        self._set_waist_ownership(False)
        for thread in (self.initialize_thread, self.move_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)
