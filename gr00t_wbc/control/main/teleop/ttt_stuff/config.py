from dataclasses import dataclass
import math

from gr00t_wbc.control.main.teleop.configs.configs import ControlLoopConfig


@dataclass
class TTTConfig(ControlLoopConfig):
    offline: bool = False
    """Use saved camera captures instead of the live ROS camera."""


def configure_ttt_config(config: TTTConfig) -> TTTConfig:
    """Apply the control settings required by the tic-tac-toe task."""
    config.enable_waist = True
    config.enable_gravity_compensation = True
    config.gravity_compensation_joints = ["right_arm"]
    config.upper_body_joint_speed = math.radians(10.0)
    return config
