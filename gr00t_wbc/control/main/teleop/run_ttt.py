import threading
import time

import tyro

from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from gr00t_wbc.control.main.teleop.run_g1_control_loop import main as run_g1_control_loop
from gr00t_wbc.control.main.teleop.ttt_stuff.controller import Controller
from gr00t_wbc.control.main.teleop.ttt_stuff.engine import move_code
from gr00t_wbc.control.main.teleop.ttt_stuff.planner import (
    BoardPlanner,
    board_corners_from_xml,
    corners_in_pelvis,
)
from gr00t_wbc.control.utils.ros_utils import ROSMsgPublisher, ROSMsgSubscriber

BOARD_STATE = "000000000"


def main(config: ControlLoopConfig):
    planner = BoardPlanner()
    controller = Controller()
    corners_world = board_corners_from_xml()
    publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
    move_thread = None

    def send(joints, duration):
        publisher.publish(
            {"target_upper_body_pose": joints, "target_time": time.monotonic() + duration}
        )

    def run_move():
        board = BOARD_STATE
        move = move_code(board)
        if move is None:
            print("No legal move available.")
            return

        state = state_subscriber.get_msg()
        floating_base_pose = None if state is None else state.get("floating_base_pose")
        corners = corners_in_pelvis(corners_world, floating_base_pose)
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
        print("Tic-tac-toe move complete", flush=True)

    def start_move():
        nonlocal move_thread
        if move_thread is not None and move_thread.is_alive():
            print("A tic-tac-toe move is already running", flush=True)
            return
        move_thread = threading.Thread(target=run_move, daemon=True)
        move_thread.start()

    def relax_arms():
        send(controller.joints, 2.0)

    run_g1_control_loop(
        config,
        program_action=start_move,
        startup_action=relax_arms,
    )


if __name__ == "__main__":
    main(tyro.cli(ControlLoopConfig))
