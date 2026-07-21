import json
import threading
import time
from pathlib import Path

import numpy as np

import tyro

from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC
from gr00t_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from gr00t_wbc.control.main.teleop.run_g1_control_loop import main as run_g1_control_loop
from gr00t_wbc.control.main.teleop.ttt_stuff.controller import Controller
from gr00t_wbc.control.main.teleop.ttt_stuff.engine import move_code
from gr00t_wbc.control.main.teleop.ttt_stuff.planner import (
    BoardPlanner,
)
from gr00t_wbc.control.utils.ros_utils import ROSMsgPublisher

LOCALIZATION = Path(__file__).resolve().parents[4] / "camera_captures/board_localization.json"
WAIST_YAW = np.deg2rad(30.0)
WAIST_DURATION = 5.0


def main(config: ControlLoopConfig):
    if config.env_type != "sim":
        raise ValueError("This saved-camera run_ttt path is simulation-only")
    config.enable_waist = True
    planner = BoardPlanner()
    controller = Controller(WAIST_YAW)
    publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    move_thread = None

    def send(joints, duration, control_waist=False):
        goal = {"target_upper_body_pose": joints, "target_time": time.monotonic() + duration}
        if control_waist:
            goal["navigate_cmd"] = np.zeros(3)
        publisher.publish(goal)

    def run_move():
        data = json.loads(LOCALIZATION.read_text())
        board = data["board_state"]
        corners = np.array([
            data["corner_cell_centers"][name] for name in ("p1", "p3", "p7", "p9")
        ])
        move = move_code(board)
        if move is None:
            print("No legal move available.")
            return

        print("Turning waist 40 degrees left", flush=True)
        finish = time.monotonic() + WAIST_DURATION
        while time.monotonic() < finish:
            send(controller.turn_left(), finish - time.monotonic(), control_waist=True)
            time.sleep(0.1)
        commands = controller.commands(planner.plan_move(board, move, corners))
        print(f"Board {board}: playing {move}", flush=True)

        for command in commands:
            if command[0] == "move":
                duration = 2.5
                send(command[1], duration)
            else:
                duration = 0.9
                send(controller.gripper(command[1]), duration)
            time.sleep(duration)
        print("Returning waist to neutral", flush=True)
        finish = time.monotonic() + WAIST_DURATION
        while time.monotonic() < finish:
            send(controller.neutral(), finish - time.monotonic(), control_waist=True)
            time.sleep(0.1)
        print("Tic-tac-toe move complete", flush=True)

    def start_move():
        nonlocal move_thread
        if move_thread is not None and move_thread.is_alive():
            print("A tic-tac-toe move is already running", flush=True)
            return
        move_thread = threading.Thread(target=run_move, daemon=True)
        move_thread.start()

    run_g1_control_loop(
        config,
        program_action=start_move,
    )


if __name__ == "__main__":
    main(tyro.cli(ControlLoopConfig))
