from dataclasses import dataclass
import time

import tyro

from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.main.teleop.movement.controller import (
    MovementController,
    UnreachableTarget,
)
from gr00t_wbc.control.main.teleop.movement.planner import BoardPlanner
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber


@dataclass
class ProgramConfig:
    board: str = "000000000"
    cell: int = 5
    enable_waist: bool = False
    state_timeout: float = 5.0


def wait_for_state(subscriber, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = subscriber.get_msg()
        if state is not None and "q" in state and "floating_base_pose" in state:
            return state
        time.sleep(0.02)
    raise TimeoutError("no robot state received; is run_g1_control_loop.py running?")


def main(config: ProgramConfig):
    ros = ROSManager(node_name="BoardMovementProgram")
    publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
    try:
        state = wait_for_state(state_subscriber, config.state_timeout)
        plan = BoardPlanner().plan(config.board, config.cell, state["floating_base_pose"])
        commands = MovementController(enable_waist=config.enable_waist).commands(plan, state["q"])
        start_time = time.monotonic() + 0.1
        elapsed = 0.0
        target_times = []
        for command in commands:
            elapsed += command.duration
            target_times.append(start_time + elapsed)
            print(f"planned: {command.name} at +{elapsed:.1f}s", flush=True)
        goal = {
            "target_upper_body_pose": [command.upper_body_pose for command in commands],
            "target_time": target_times,
        }
        if config.enable_waist:
            goal["preserve_upper_body_waist_pitch"] = True
        publisher.publish(goal)
        print(f"trajectory sent for board {config.board}, cell {config.cell}", flush=True)
        time.sleep(0.1)
    except (TimeoutError, ValueError, UnreachableTarget) as error:
        print(f"movement cancelled: {error}", flush=True)
    finally:
        ros.shutdown()


if __name__ == "__main__":
    main(tyro.cli(ProgramConfig))
