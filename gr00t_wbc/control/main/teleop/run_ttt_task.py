"""Tic-tac-toe task, vision, and IK process isolated from the WBC loop."""

import time

import tyro

from gr00t_wbc.control.main.teleop.ttt_stuff.config import (
    TTTConfig,
    configure_ttt_config,
)
from gr00t_wbc.control.main.teleop.ttt_stuff.program import TTTProgram
from gr00t_wbc.control.utils.keyboard_dispatcher import KeyboardListenerSubscriber
from gr00t_wbc.control.utils.ros_utils import ROSManager


def main(config: TTTConfig) -> None:
    """Run camera, detection, planning, and IK outside the control process."""
    config = configure_ttt_config(config)
    ros_manager = ROSManager(node_name="ttt_task")
    keyboard = KeyboardListenerSubscriber()
    program = TTTProgram(config)

    try:
        print("TTT task waiting for the WBC state stream", flush=True)
        while ros_manager.ok() and not program.wait_for_state(timeout=0.5):
            pass
        if not ros_manager.ok():
            return

        print("WBC state stream ready; starting arm initialization", flush=True)
        program.initialize()

        while ros_manager.ok():
            key = keyboard.read_msg()
            if key == "p":
                program.start_move()
            elif key in ("space", " "):
                program.continue_move()
            time.sleep(0.01)
    except ros_manager.exceptions() as error:
        print(f"TTT task interrupted: {error}", flush=True)
    finally:
        print("Stopping TTT task process", flush=True)
        program.shutdown()
        ros_manager.shutdown()


if __name__ == "__main__":
    main(tyro.cli(TTTConfig))
