import threading
import time
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


def main(config: ControlLoopConfig):
    if config.env_type != "sim":
        raise ValueError("This saved-camera run_ttt path is simulation-only")
    config.enable_waist = True
    planner = BoardPlanner()
    controller = Controller(np.deg2rad(20.0))
    publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
    move_thread = None
    board_world = None

    def send(joints, duration, corners=None):
        goal = {
            "target_upper_body_pose": joints,
            "target_time": time.monotonic() + duration,
            "preserve_upper_body_waist_yaw": True,
        }
        if corners is not None:
            goal["ttt_corners"] = corners.tolist()
        publisher.publish(goal)

    def live_corners():
        state = state_subscriber.get_msg()
        if state is None or state.get("floating_base_pose") is None:
            raise RuntimeError("Pelvis state unavailable")
        pose = np.asarray(state["floating_base_pose"], dtype=float)
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]])
        return state, rotation.inv().apply(board_world - pose[:3])

    def run_move():
        nonlocal board_world
        frame, radial, intrinsics = capture()
        data = localize(detect(frame), radial, intrinsics)
        board = data["board_state"]
        corners = np.array([
            data["corner_cell_centers"][name] for name in ("p1", "p3", "p7", "p9")
        ])
        state = state_subscriber.get_msg()
        if state is None or state.get("floating_base_pose") is None:
            print("No pelvis pose available; move cancelled", flush=True)
            return
        before = np.asarray(state["floating_base_pose"], dtype=float)
        rotation_before = Rotation.from_quat(before[[4, 5, 6, 3]])
        board_world = rotation_before.apply(corners) + before[:3]

        print("Turning waist 20 degrees left", flush=True)
        send(controller.turn_left(), 5.0)
        time.sleep(5.0)
        state_after, corners = live_corners()
        after = np.asarray(state_after["floating_base_pose"], dtype=float)
        rotation_after = Rotation.from_quat(after[[4, 5, 6, 3]])
        yaw_change = (rotation_after.inv() * rotation_before).as_euler("xyz")[2]
        publisher.publish({"ttt_corners": corners.tolist()})
        time.sleep(0.2)
        print(
            f"Pelvis correction: yaw={np.degrees(yaw_change):.2f} deg, "
            f"translation={np.round(after[:3] - before[:3], 4)} m",
            flush=True,
        )
        move = move_code(board)
        if move is None:
            print("No legal move available.")
            return

        commands = planner.plan_move(board, move, corners)
        print(f"Board {board}: playing {move}", flush=True)

        active_pose = None
        smooth_corners = corners.copy()
        smooth_joints = controller.turn_left()
        for index, command in enumerate(commands):
            if command[0] == "move":
                active_pose = index
                duration = 4.0
            else:
                duration = 1.5
                controller.gripper(command[1])
            deadline = None
            while deadline is None or time.monotonic() < deadline:
                state, raw_corners = live_corners()
                smooth_corners += 0.12 * (raw_corners - smooth_corners)
                velocity = np.asarray(state["floating_base_vel"], dtype=float)
                if (np.linalg.norm(velocity[:3]) > 0.15 or
                        np.linalg.norm(velocity[3:]) > 0.5):
                    print("Unsafe pelvis speed; arm move cancelled", flush=True)
                    send(controller.neutral(), 5.0, smooth_corners)
                    return
                live_plan = planner.plan_move(board, move, smooth_corners)
                target = live_plan[active_pose]
                ik_joints = controller.ik(target[1], target[2])
                smooth_joints += 0.18 * (ik_joints - smooth_joints)
                if deadline is None:
                    deadline = time.monotonic() + duration
                send(smooth_joints, 1.0, smooth_corners)
                time.sleep(0.2)
        print("Returning waist to neutral", flush=True)
        neutral_joints = controller.neutral()
        finish = time.monotonic() + 5.0
        while time.monotonic() < finish:
            _, raw_corners = live_corners()
            smooth_corners += 0.12 * (raw_corners - smooth_corners)
            smooth_joints += 0.12 * (neutral_joints - smooth_joints)
            send(smooth_joints, 1.0, smooth_corners)
            time.sleep(0.2)
        send(neutral_joints, 2.0, smooth_corners)
        time.sleep(2.0)
        print("Tic-tac-toe move complete", flush=True)

    def start_move():
        nonlocal move_thread
        if move_thread is not None and move_thread.is_alive():
            print("A tic-tac-toe move is already running", flush=True)
            return
        move_thread = threading.Thread(target=run_move, daemon=True)
        move_thread.start()

    def relax_arms():
        send(controller.neutral(), 2.0)

    run_g1_control_loop(
        config,
        program_action=start_move,
        startup_action=relax_arms,
    )


if __name__ == "__main__":
    main(tyro.cli(ControlLoopConfig))
