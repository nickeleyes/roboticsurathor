from dataclasses import dataclass
import math
from typing import Literal

from gr00t_wbc.control.main.teleop.configs.configs import ControlLoopConfig


@dataclass
class TTTConfig(ControlLoopConfig):
    offline: bool = False
    """Use fixed board geometry without loading the camera or vision detector."""
    arm_side: Literal["left", "right"] = "right"
    """Arm used for tic-tac-toe."""


def configure_ttt_config(config: TTTConfig) -> TTTConfig:
    """Apply the control settings required by the tic-tac-toe task."""
    config.enable_waist = True
    config.enable_gravity_compensation = True
    config.gravity_compensation_joints = [f"{config.arm_side}_arm"]
    config.upper_body_joint_speed = math.radians(10.0)
    return config
