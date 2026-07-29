from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import threading
import time
import traceback

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import tyro

from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from gr00t_wbc.control.main.teleop.run_g1_control_loop import main as run_g1_control_loop
from gr00t_wbc.control.main.teleop.ttt_stuff.controller import Controller
from gr00t_wbc.control.main.teleop.ttt_stuff.engine import move_code
from gr00t_wbc.control.main.teleop.ttt_stuff.localization import localize
from gr00t_wbc.control.main.teleop.ttt_stuff.planner import (
    BoardPlanner,
)
from gr00t_wbc.control.main.teleop.ttt_stuff.ros2capture import capture
from gr00t_wbc.control.main.teleop.ttt_stuff.vision import detect
from gr00t_wbc.control.utils.ros_utils import ROSMsgPublisher, ROSMsgSubscriber

POSE_DURATION = 5.0
WAYPOINT_DURATION = 6.0
KEEPALIVE_PERIOD = 0.4
SAVED_CAPTURE_DIR = Path(__file__).resolve().parents[4] / "camera_captures"
RIGHT_ARM_DOWN = np.array(
    [0.0, np.deg2rad(-30.0), 0.0, np.pi / 2, 0.0, 0.0, 0.0]
)
RIGHT_ARM_SIDE = np.array(
    [0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0, 0.0]
)
RIGHT_ARM_SIDE_BENT = np.array(
    [0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0]
)


@dataclass
class TTTConfig(ControlLoopConfig):
    offline: bool = False
    """Use saved camera_captures instead of the live ROS camera."""


def computation_worker(requests, results):
    """Run board detection and one-time arm-only IK away from robot control."""
    import torch

    try:
        os.nice(10)
    except OSError:
        pass
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    planner = BoardPlanner()
    while True:
        request = requests.get()
        if request is None:
            return
        try:
            if request["operation"] == "vision":
                data = localize(
                    detect(request["frame"]),
                    request["radial"],
                    request["intrinsics"],
                )
                board = data["board_state"]
                corners = np.array([
                    data["corner_cell_centers"][name]
                    for name in ("p1", "p3", "p7", "p9")
                ])
                move = move_code(board)
                plan = (
                    planner.plan_move(board, move, corners)
                    if move is not None
                    else []
                )
                results.put({
                    "ok": True,
                    "board": board,
                    "corners": corners,
                    "move": move,
                    "target_positions": [command[1] for command in plan],
                    "target_rotations": [command[2] for command in plan],
                    "target_heights": [command[3] for command in plan],
                    "target_names": [command[4] for command in plan],
                })
                continue

            raise ValueError(f"Unknown worker operation: {request['operation']}")
        except Exception:
            results.put({"ok": False, "error": traceback.format_exc()})


def main(config: TTTConfig):
    config.enable_waist = True
    controller = Controller(0.0)
    process_context = multiprocessing.get_context("spawn")
    worker_requests = process_context.Queue()
    worker_results = process_context.Queue()
    worker = process_context.Process(
        target=computation_worker,
        args=(worker_requests, worker_results),
        daemon=True,
    )
    worker.start()
    publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
    move_thread = None
    initialization_done = threading.Event()
    continue_event = threading.Event()

    def arm_pose(right_arm):
        joints = controller.joints.copy()
        joints[controller.waist_yaw] = 0.0
        joints[controller.right_arm] = right_arm
        return joints

    def send(
        joints,
        duration,
        use_groot_torso=False,
        ik_target_world=None,
    ):
        goal = {
            "target_upper_body_pose": joints,
            "target_time": time.monotonic() + duration,
            "preserve_upper_body_waist_yaw": True,
        }
        if use_groot_torso:
            goal["navigate_cmd"] = np.zeros(3, dtype=np.float32)
        if ik_target_world is not None:
            goal["ttt_ik_target"] = np.asarray(
                ik_target_world,
                dtype=float,
            ).tolist()
        publisher.publish(goal)

    def keepalive():
        publisher.publish({
            "navigate_cmd": np.zeros(3, dtype=np.float32),
            "preserve_upper_body_waist_yaw": True,
        })

    def execute_target(joints, duration, ik_target_world=None):
        send(
            joints,
            duration,
            use_groot_torso=True,
            ik_target_world=ik_target_world,
        )
        deadline = time.monotonic() + duration
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(KEEPALIVE_PERIOD, remaining))
            if time.monotonic() < deadline:
                keepalive()

    def wait_for_space(message):
        continue_event.clear()
        print(f"{message} Press SPACE to continue.", flush=True)
        while not continue_event.wait(timeout=KEEPALIVE_PERIOD):
            keepalive()

    def compute(request, description):
        worker_requests.put(request)
        result = worker_results.get()
        if not result["ok"]:
            raise RuntimeError(f"{description} failed in worker:\n{result['error']}")
        return result

    def get_robot_state(timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = state_subscriber.get_msg()
            if state is not None and state.get("q") is not None:
                return state
            time.sleep(0.02)
        return None

    def run_move():
        print("Capturing the tic-tac-toe board", flush=True)
        try:
            if config.offline:
                frame = cv2.imread(str(SAVED_CAPTURE_DIR / "color_rgb.png"))
                if frame is None:
                    raise FileNotFoundError(SAVED_CAPTURE_DIR / "color_rgb.png")
                radial = np.load(SAVED_CAPTURE_DIR / "radial_distance_m.npy")
                intrinsics = json.loads(
                    (SAVED_CAPTURE_DIR / "camera_intrinsics.json").read_text(
                        encoding="utf-8"
                    )
                )
                print(
                    f"Using saved camera capture: {SAVED_CAPTURE_DIR}",
                    flush=True,
                )
            else:
                frame, radial, intrinsics = capture()
            state = get_robot_state()
            if state is None:
                print("Robot joint state unavailable; move cancelled", flush=True)
                return
            print("Detecting the board in the worker process", flush=True)
            vision = compute(
                {
                    "operation": "vision",
                    "frame": frame,
                    "radial": radial,
                    "intrinsics": intrinsics,
                },
                "Board detection",
            )
        except Exception as error:
            print(f"Board capture cancelled: {error}", flush=True)
            return
        board = vision["board"]
        corners = vision["corners"]
        move = vision["move"]
        marker_goal = {}
        floating_base_pose = state.get("floating_base_pose")
        pelvis_rotation = None
        pelvis_position = None
        board_center_world = None
        if floating_base_pose is not None:
            floating_base_pose = np.asarray(floating_base_pose, dtype=float)
            pelvis_rotation = Rotation.from_quat(
                floating_base_pose[[4, 5, 6, 3]]
            ).as_matrix()
            pelvis_position = floating_base_pose[:3]
            reference_world = (
                corners @ pelvis_rotation.T + pelvis_position
            )
            board_center_world = np.mean(reference_world, axis=0)
            marker_goal["ttt_reference_corners_world"] = reference_world.tolist()
            print(
                "Board corners (world frame):\n"
                f"{np.round(reference_world, 4)}\n"
                f"Board center (world frame): {np.round(board_center_world, 4)}",
                flush=True,
            )
        if marker_goal:
            publisher.publish(marker_goal)
        if move is None:
            print("No legal move available.")
            return

        print(f"Board {board}: playing {move}", flush=True)
        print("Executing arm-only IK from the displayed waypoint targets", flush=True)
        wait_for_space("Board captured; waypoint 1 is ready.")

        waypoint_data = zip(
            vision["target_positions"],
            vision["target_rotations"],
            vision["target_heights"],
            vision["target_names"],
        )
        target_count = len(vision["target_positions"])
        for index, (position, rotation, height, name) in enumerate(waypoint_data):
            position = np.asarray(position, dtype=float)
            rotation = np.asarray(rotation, dtype=float)
            if pelvis_rotation is not None:
                display_position = pelvis_rotation @ position + pelvis_position
                frame_name = "world"
            else:
                display_position = position
                frame_name = "pelvis"
            joints = controller.ik(position, rotation).copy()
            details = (
                f"Waypoint {index + 1}/{target_count}: "
                f"{name} {height}, target ({frame_name})="
                f"{np.round(display_position, 4)}."
            )
            if name == "pos5" and board_center_world is not None:
                details += (
                    f" Board center (world)={np.round(board_center_world, 4)}, "
                    f"offset={np.round(display_position - board_center_world, 4)}."
                )
            print(details, flush=True)
            execute_target(
                joints,
                WAYPOINT_DURATION,
                ik_target_world=(
                    display_position if pelvis_rotation is not None else None
                ),
            )
            print(f"Completed planned arm step {index + 1}", flush=True)
            if index + 1 < target_count:
                wait_for_space(
                    f"Waypoint {index + 1} complete; "
                    f"waypoint {index + 2} is ready."
                )

        wait_for_space("Tic-tac-toe move complete.")

        print("Returning to shoulder roll 90 degrees with elbow at 0", flush=True)
        execute_target(arm_pose(RIGHT_ARM_SIDE_BENT), POSE_DURATION)
        print("Tic-tac-toe move complete", flush=True)

    def start_move():
        nonlocal move_thread
        if not initialization_done.is_set():
            print("Arm initialization is still running; wait for it to finish", flush=True)
            return
        if move_thread is not None and move_thread.is_alive():
            print("A tic-tac-toe move is already running", flush=True)
            return
        publisher.publish({"ttt_lock_feet": True})
        move_thread = threading.Thread(target=run_move, daemon=True)
        move_thread.start()

    def continue_move():
        continue_event.set()

    def run_initialization_sequence():
        print("Initialization 1/3: shoulder roll 30, elbow 90", flush=True)
        send(arm_pose(RIGHT_ARM_DOWN), POSE_DURATION)
        time.sleep(POSE_DURATION)
        print("Initialization 2/3: right shoulder roll 90, elbow 90", flush=True)
        send(arm_pose(RIGHT_ARM_SIDE), POSE_DURATION)
        time.sleep(POSE_DURATION)
        print("Initialization 3/3: right shoulder roll 90, elbow 0", flush=True)
        send(arm_pose(RIGHT_ARM_SIDE_BENT), POSE_DURATION)
        time.sleep(POSE_DURATION)
        initialization_done.set()
        print("Arm initialization complete; press p to capture the board", flush=True)

    def start_initialization_sequence():
        threading.Thread(
            target=run_initialization_sequence,
            daemon=True,
        ).start()

    try:
        run_g1_control_loop(
            config,
            program_action=start_move,
            program_continue_action=continue_move,
            startup_action=start_initialization_sequence,
        )
    finally:
        try:
            worker_requests.put_nowait(None)
        except (OSError, ValueError):
            pass
        try:
            worker.join(timeout=2.0)
        except KeyboardInterrupt:
            pass
        if worker.is_alive():
            worker.terminate()
            try:
                worker.join(timeout=1.0)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main(tyro.cli(TTTConfig))
