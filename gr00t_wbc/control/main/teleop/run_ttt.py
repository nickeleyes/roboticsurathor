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
RIGHT_ARM_JOINT_NAMES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
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
    config.enable_gravity_compensation = True
    config.gravity_compensation_joints = ["right_arm"]
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
            "preserve_upper_body_waist_roll": True,
            "preserve_upper_body_waist_pitch": True,
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
            "preserve_upper_body_waist_roll": True,
            "preserve_upper_body_waist_pitch": True,
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

    def grip_fk_in_pelvis(full_q):
        full_q = np.asarray(full_q, dtype=float)
        controller.model.cache_forward_kinematics(full_q, auto_clip=False)
        return controller.model.frame_placement(controller.site).translation.copy()

    def print_arm_angles(label, angles):
        angles_deg = np.rad2deg(np.asarray(angles, dtype=float))
        print(f"{label} right-arm angles (degrees):", flush=True)
        for joint_name, angle in zip(RIGHT_ARM_JOINT_NAMES, angles_deg):
            print(f"  {joint_name}: {angle:.3f}", flush=True)

    def print_waist_angles(label, full_q):
        angles = [
            full_q[controller.model.dof_index(name)]
            for name in WAIST_JOINT_NAMES
        ]
        print(
            f"{label} waist angles (degrees): "
            f"yaw={np.rad2deg(angles[0]):.3f}, "
            f"roll={np.rad2deg(angles[1]):.3f}, "
            f"pitch={np.rad2deg(angles[2]):.3f}",
            flush=True,
        )

    def base_pose_parts(base_pose):
        base_pose = np.asarray(base_pose, dtype=float)
        rotation = Rotation.from_quat(base_pose[[4, 5, 6, 3]])
        return base_pose[:3], rotation.as_matrix(), rotation.as_euler(
            "xyz", degrees=True
        )

    def print_base_delta(label, first_position, first_rotation, second_position, second_rotation):
        rotation_delta = Rotation.from_matrix(
            first_rotation.T @ second_rotation
        )
        print(
            f"{label}: translation_world="
            f"{np.round(second_position - first_position, 5)} m, "
            f"rotation_angle={np.rad2deg(rotation_delta.magnitude()):.4f} deg, "
            f"rotation_xyz={np.round(rotation_delta.as_euler('xyz', degrees=True), 4)} deg",
            flush=True,
        )

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
            ik_position = position
            ik_rotation = rotation
            solve_pelvis_rotation = pelvis_rotation
            solve_pelvis_position = pelvis_position
            solve_pelvis_rpy = (
                Rotation.from_matrix(pelvis_rotation).as_euler(
                    "xyz", degrees=True
                )
                if pelvis_rotation is not None
                else None
            )
            if pelvis_rotation is not None:
                display_position = pelvis_rotation @ position + pelvis_position
                display_rotation = pelvis_rotation @ rotation
                frame_name = "world"

                solve_state = get_robot_state()
                solve_base_pose = (
                    None
                    if solve_state is None
                    else solve_state.get("floating_base_pose")
                )
                if solve_base_pose is not None:
                    (
                        solve_pelvis_position,
                        solve_pelvis_rotation,
                        solve_pelvis_rpy,
                    ) = base_pose_parts(solve_base_pose)
                    ik_position = (
                        solve_pelvis_rotation.T
                        @ (display_position - solve_pelvis_position)
                    )
                    ik_rotation = (
                        solve_pelvis_rotation.T @ display_rotation
                    )
            else:
                display_position = position
                frame_name = "pelvis"
            joints = controller.ik(ik_position, ik_rotation).copy()
            requested_full_q = controller.model.default_body_pose.copy()
            requested_full_q[
                controller.model.get_joint_group_indices("upper_body")
            ] = joints
            requested_grip_pelvis = grip_fk_in_pelvis(requested_full_q)
            requested_grip_world = (
                solve_pelvis_rotation @ requested_grip_pelvis
                + solve_pelvis_position
                if solve_pelvis_rotation is not None
                else requested_grip_pelvis
            )
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
            print(
                f"Planner target (captured pelvis)={np.round(position, 5)} m; "
                f"corrected IK target (solve pelvis)={np.round(ik_position, 5)} m",
                flush=True,
            )
            if solve_pelvis_rotation is not None:
                print(
                    f"Captured pelvis world position={np.round(pelvis_position, 5)} m; "
                    f"solve pelvis world position={np.round(solve_pelvis_position, 5)} m; "
                    f"solve pelvis RPY={np.round(solve_pelvis_rpy, 4)} deg",
                    flush=True,
                )
                print_base_delta(
                    "Captured-to-solve pelvis delta",
                    pelvis_position,
                    pelvis_rotation,
                    solve_pelvis_position,
                    solve_pelvis_rotation,
                )
            print_arm_angles("Requested", joints[controller.right_arm])
            print_waist_angles("Requested", requested_full_q)
            print(
                "Requested-joint FK grip (solve pelvis)="
                f"{np.round(requested_grip_pelvis, 5)} m, "
                "FK-minus-corrected-IK-target="
                f"{np.round(requested_grip_pelvis - ik_position, 5)} m",
                flush=True,
            )
            print(
                "Requested-joint FK grip "
                f"({frame_name})={np.round(requested_grip_world, 4)}, "
                f"red target={np.round(display_position, 4)}, "
                "FK-minus-red="
                f"{np.round(requested_grip_world - display_position, 4)} m",
                flush=True,
            )
            execute_target(
                joints,
                WAYPOINT_DURATION,
                ik_target_world=(
                    display_position if pelvis_rotation is not None else None
                ),
            )

            actual_state = get_robot_state()
            if actual_state is None:
                print("Actual robot state unavailable after motion.", flush=True)
            else:
                actual_q = np.asarray(actual_state["q"], dtype=float)
                if actual_q.shape != requested_full_q.shape:
                    print(
                        "Cannot run actual-joint FK: state q has shape "
                        f"{actual_q.shape}, expected {requested_full_q.shape}.",
                        flush=True,
                    )
                else:
                    right_arm_indices = controller.model.get_joint_group_indices(
                        "right_arm"
                    )
                    wbc_q = actual_state.get("action")
                    if wbc_q is None:
                        print(
                            "Final WBC q action unavailable in robot state.",
                            flush=True,
                        )
                    else:
                        wbc_q = np.asarray(wbc_q, dtype=float)
                        if wbc_q.shape != requested_full_q.shape:
                            print(
                                "Cannot inspect final WBC q: action has shape "
                                f"{wbc_q.shape}, expected {requested_full_q.shape}.",
                                flush=True,
                            )
                        else:
                            print_arm_angles(
                                "Final WBC command",
                                wbc_q[right_arm_indices],
                            )
                            print_waist_angles("Final WBC command", wbc_q)
                    print_arm_angles("Actual", actual_q[right_arm_indices])
                    print_waist_angles("Actual", actual_q)
                    print(
                        "Actual-minus-WBC right-arm angle error (degrees)="
                        f"{np.round(np.rad2deg(actual_q[right_arm_indices] - wbc_q[right_arm_indices]), 4)}"
                        if wbc_q is not None
                        and wbc_q.shape == requested_full_q.shape
                        else "Actual-minus-WBC angle error unavailable",
                        flush=True,
                    )
                    actual_grip_pelvis = grip_fk_in_pelvis(actual_q)
                    actual_base_pose = actual_state.get("floating_base_pose")
                    if actual_base_pose is not None:
                        (
                            actual_pelvis_position,
                            actual_pelvis_rotation,
                            actual_pelvis_rpy,
                        ) = base_pose_parts(actual_base_pose)
                        actual_grip_world = (
                            actual_pelvis_rotation @ actual_grip_pelvis
                            + actual_pelvis_position
                        )
                        actual_frame_name = "world"
                        print(
                            "Final pelvis world position="
                            f"{np.round(actual_pelvis_position, 5)} m, "
                            f"RPY={np.round(actual_pelvis_rpy, 4)} deg",
                            flush=True,
                        )
                        print_base_delta(
                            "Captured-to-final pelvis delta",
                            pelvis_position,
                            pelvis_rotation,
                            actual_pelvis_position,
                            actual_pelvis_rotation,
                        )
                        print_base_delta(
                            "Solve-to-final pelvis delta",
                            solve_pelvis_position,
                            solve_pelvis_rotation,
                            actual_pelvis_position,
                            actual_pelvis_rotation,
                        )
                    else:
                        actual_grip_world = actual_grip_pelvis
                        actual_frame_name = "pelvis"
                    if (
                        wbc_q is not None
                        and wbc_q.shape == requested_full_q.shape
                    ):
                        wbc_grip_pelvis = grip_fk_in_pelvis(wbc_q)
                        if actual_base_pose is not None:
                            wbc_grip_world = (
                                actual_pelvis_rotation @ wbc_grip_pelvis
                                + actual_base_pose[:3]
                            )
                        else:
                            wbc_grip_world = wbc_grip_pelvis
                        print(
                            "Final-WBC-command FK grip (final pelvis)="
                            f"{np.round(wbc_grip_pelvis, 5)} m",
                            flush=True,
                        )
                        print(
                            "Final-WBC-command FK grip "
                            f"({actual_frame_name})="
                            f"{np.round(wbc_grip_world, 4)}, "
                            f"red target={np.round(display_position, 4)}, "
                            "FK-minus-red="
                            f"{np.round(wbc_grip_world - display_position, 4)} m",
                            flush=True,
                        )
                    print(
                        "Actual-joint FK grip (final pelvis)="
                        f"{np.round(actual_grip_pelvis, 5)} m",
                        flush=True,
                    )
                    print(
                        "Actual-joint FK grip "
                        f"({actual_frame_name})={np.round(actual_grip_world, 4)}, "
                        f"red target={np.round(display_position, 4)}, "
                        "FK-minus-red="
                        f"{np.round(actual_grip_world - display_position, 4)} m",
                        flush=True,
                    )
                    tau_est = actual_state.get("tau_est")
                    if tau_est is not None:
                        tau_est = np.asarray(tau_est, dtype=float)
                        if tau_est.shape == requested_full_q.shape:
                            gravity_torque = (
                                controller.model.compute_gravity_compensation_torques(
                                    actual_q,
                                    joint_groups=["right_arm"],
                                    auto_clip=False,
                                )
                            )
                            print(
                                "Actual estimated right-arm torque (Nm)="
                                f"{np.round(tau_est[right_arm_indices], 4)}",
                                flush=True,
                            )
                            print(
                                "Pinocchio right-arm gravity torque (Nm)="
                                f"{np.round(gravity_torque[right_arm_indices], 4)}",
                                flush=True,
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
