from dataclasses import dataclass

import numpy as np
import tyro

from gr00t_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from gr00t_wbc.control.main.teleop.run_g1_control_loop import main as run_g1_control_loop
from gr00t_wbc.control.main.teleop.ttt_stuff.program import TTTProgram


@dataclass
class TTTConfig(ControlLoopConfig):
    offline: bool = False
    """Use saved camera captures instead of the live ROS camera."""


def main(config: TTTConfig):
    config.enable_waist = True
    config.enable_gravity_compensation = True
    config.gravity_compensation_joints = ["right_arm"]
    config.upper_body_joint_speed = np.deg2rad(10.0)
    program = TTTProgram(config)
    run_g1_control_loop(
        config,
        program_action=program.start_move,
        program_continue_action=program.continue_move,
        startup_action=program.initialize,
    )


if __name__ == "__main__":
    main(tyro.cli(TTTConfig))
