import multiprocessing
import os
import threading
import time
import traceback

import numpy as np

import tyro

from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from gr00t_wbc.control.main.teleop.run_g1_control_loop import main as run_g1_control_loop
from gr00t_wbc.control.main.teleop.ttt_stuff.controller import Controller
from gr00t_wbc.control.main.teleop.ttt_stuff.engine import move_code
from gr00t_wbc.control.main.teleop.ttt_stuff.localization import localize
from gr00t_wbc.control.main.teleop.ttt_stuff.pelvis_tracking import PelvisTracker
from gr00t_wbc.control.main.teleop.ttt_stuff.planner import (
    BoardPlanner,
)
from gr00t_wbc.control.main.teleop.ttt_stuff.ros2capture import capture
from gr00t_wbc.control.main.teleop.ttt_stuff.vision import detect
from gr00t_wbc.control.utils.ros_utils import ROSMsgPublisher, ROSMsgSubscriber

POSE_DURATION = 5.0
POSE_BLEND = 0.18
SETTLE_DURATION = 0.5
YAW_KEY_COUNT = 4
YAW_KEY_PERIOD = 1.0


def computation_worker(requests, results):
    """Run vision, FK, and IK away from the robot-control process."""
    import cv2
    import torch

    try:
        os.nice(10)
    except OSError:
        pass
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
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
                results.put({
                    "ok": True,
                    "board": board, 
                    "corners": corners,
                    "move": move_code(board),
                })
                continue

            tracker = PelvisTracker(translation_tolerance=0.06)
            tracker.capture_initial(request["initial_q"])
            tracker.capture_final(request["final_q"])
            tracking = tracker.pelvis_delta()
            corners = tracker.transform_points(request["corners"])

            controller = Controller(request["waist_yaw"])
            commands = BoardPlanner().plan_move(
                request["board"],
                request["move"],
                corners,
            )
            joint_targets = []
            for command in commands:
                if command[0] == "move":
                    # Let the iterative solver settle fully in the worker.
                    for _ in range(4):
                        joints = controller.ik(command[1], command[2])
                    joint_targets.append({
                        "joints": joints,
                        "duration": 4.0,
                    })
                else:
                    joint_targets.append({
                        "joints": controller.gripper(command[1]).copy(),
                        "duration": 1.5,
                    })
            results.put({
                "ok": True,
                "corners": corners,
                "tracking": tracking,
                "joint_targets": joint_targets,
            })
        except Exception:
            results.put({"ok": False, "error": traceback.format_exc()})


def main(config: ControlLoopConfig):
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
    initialization_thread = None
    initialization_done = threading.Event()
    policy_key_event = 0
    continue_event = threading.Event()

    def send(joints, duration, corners=None, preserve_waist=False):
        goal = {
            "target_upper_body_pose": joints,
            "target_time": time.monotonic() + duration,
            "preserve_upper_body_waist_yaw": preserve_waist,
        }
        if corners is not None:
            goal["ttt_corners"] = corners.tolist()
        publisher.publish(goal)

    def press_policy_key(key, count):
        nonlocal policy_key_event
        for _ in range(count):
            policy_key_event += 1
            publisher.publish({
                "policy_key": key,
                "policy_key_event": policy_key_event,
            })
            time.sleep(YAW_KEY_PERIOD)

    def wait_for_space(message):
        continue_event.clear()
        print(f"{message} Press SPACE to continue.", flush=True)
        continue_event.wait()

    def compute(request, description):
        worker_requests.put(request)
        result = worker_results.get()
        if not result["ok"]:
            raise RuntimeError(f"{description} failed in worker:\n{result['error']}")
        return result

    def run_move():
        print("Capturing the tic-tac-toe board", flush=True)
        try:
            frame, radial, intrinsics = capture()
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
        state = state_subscriber.get_msg()
        if state is None or state.get("q") is None:
            print("Robot joint state unavailable; move cancelled", flush=True)
            return
        publisher.publish({"ttt_corners": corners.tolist()})
        if move is None:
            print("No legal move available.")
            return

        wait_for_space("Board captured.")

        state = state_subscriber.get_msg()
        if state is None or state.get("q") is None:
            print("Pre-rotation joint state unavailable; move cancelled", flush=True)
            return
        initial_q = np.asarray(state["q"], dtype=float).copy()
        print("Sending GR00T yaw key 8 four times", flush=True)
        press_policy_key("8", YAW_KEY_COUNT)
        time.sleep(POSE_DURATION)

        wait_for_space("Waist rotation complete.")

        state = state_subscriber.get_msg()
        if state is None or state.get("q") is None:
            print("Final leg joint state unavailable; move cancelled", flush=True)
            return
        waist_index = controller.model.dof_index("waist_yaw_joint")
        achieved_waist_yaw = float(np.asarray(state["q"])[waist_index])
        print(
            f"GR00T rotation settled with waist at "
            f"{np.degrees(achieved_waist_yaw):.2f} degrees",
            flush=True,
        )

        time.sleep(SETTLE_DURATION)

        state = state_subscriber.get_msg()
        if state is None or state.get("q") is None:
            print("Settled leg joint state unavailable; move cancelled", flush=True)
            return
        try:
            print("Computing pelvis correction and arm IK in worker process", flush=True)
            plan = compute(
                {
                    "operation": "plan",
                    "initial_q": initial_q,
                    "final_q": np.asarray(state["q"], dtype=float).copy(),
                    "corners": corners,
                    "board": board,
                    "move": move,
                    "waist_yaw": achieved_waist_yaw,
                },
                "Pelvis tracking and arm planning",
            )
        except Exception as error:
            print(f"Move cancelled: {error}", flush=True)
            return
        tracking = plan["tracking"]
        corners = plan["corners"]
        delta = tracking["final_pelvis_T_initial_pelvis"]
        rotation_degrees = np.degrees(
            np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
        )
        print(
            "Pelvis correction from fixed-foot FK: "
            f"translation={np.round(delta[:3, 3], 4)} m, "
            f"rotation={rotation_degrees:.2f} deg, "
            f"foot disagreement={tracking['translation_disagreement_m']:.4f} m/"
            f"{np.degrees(tracking['rotation_disagreement_rad']):.2f} deg",
            flush=True,
        )
        publisher.publish({"ttt_corners": corners.tolist()})

        smooth_joints = controller.side_t_bent_pose(achieved_waist_yaw)

        print(f"Board {board}: playing {move}", flush=True)
        print("Moving right arm to hover and executing the tic-tac-toe move", flush=True)

        for index, target in enumerate(plan["joint_targets"]):
            deadline = time.monotonic() + target["duration"]
            while time.monotonic() < deadline:
                ik_joints = target["joints"]
                smooth_joints += POSE_BLEND * (ik_joints - smooth_joints)
                send(smooth_joints, 1.0, corners)
                time.sleep(0.2)
            print(f"Completed planned arm step {index + 1}", flush=True)

        wait_for_space("Tic-tac-toe move complete.")

        print("Returning to shoulder roll 90 degrees with elbow at 0", flush=True)
        side_t = controller.side_t_bent_pose(achieved_waist_yaw)
        finish = time.monotonic() + POSE_DURATION
        while time.monotonic() < finish:
            smooth_joints += POSE_BLEND * (side_t - smooth_joints)
            send(smooth_joints, 1.0, corners)
            time.sleep(0.2)
        send(side_t, 2.0, corners)
        time.sleep(2.0)
        print("Sending GR00T yaw key 7 four times", flush=True)
        press_policy_key("7", YAW_KEY_COUNT)
        send(
            controller.side_t_bent_pose(0.0),
            POSE_DURATION,
            corners,
            preserve_waist=False,
        )
        time.sleep(POSE_DURATION)
        print("Tic-tac-toe move complete", flush=True)

    def start_move():
        nonlocal move_thread
        if not initialization_done.is_set():
            print("Arm initialization is still running; wait for it to finish", flush=True)
            return
        if move_thread is not None and move_thread.is_alive():
            print("A tic-tac-toe move is already running", flush=True)
            return
        move_thread = threading.Thread(target=run_move, daemon=True)
        move_thread.start()

    def continue_move():
        continue_event.set()

    def run_initialization_sequence():
        print("Initialization 1/3: shoulder roll 30, elbow 90", flush=True)
        send(controller.both_arms_down_pose(), POSE_DURATION)
        time.sleep(POSE_DURATION)
        print("Initialization 2/3: right shoulder roll 90, elbow 90", flush=True)
        send(controller.side_t_pose(), POSE_DURATION)
        time.sleep(POSE_DURATION)
        print("Initialization 3/3: right shoulder roll 90, elbow 0", flush=True)
        send(controller.side_t_bent_pose(), POSE_DURATION)
        time.sleep(POSE_DURATION)
        initialization_done.set()
        print("Arm initialization complete; press p to capture the board", flush=True)

    def start_initialization_sequence():
        nonlocal initialization_thread
        initialization_thread = threading.Thread(
            target=run_initialization_sequence,
            daemon=True,
        )
        initialization_thread.start()

    try:
        run_g1_control_loop(
            config,
            program_action=start_move,
            program_continue_action=continue_move,
            startup_action=start_initialization_sequence,
        )
    finally:
        worker_requests.put(None)
        worker.join(timeout=2.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)


if __name__ == "__main__":
    main(tyro.cli(ControlLoopConfig))
