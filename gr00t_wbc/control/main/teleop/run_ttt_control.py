"""Dedicated 50 Hz whole-body-control process for tic-tac-toe."""

import tyro

from gr00t_wbc.control.main.teleop.run_g1_control_loop import main as run_g1_control_loop
from gr00t_wbc.control.main.teleop.ttt_stuff.config import (
    TTTConfig,
    configure_ttt_config,
)


def main(config: TTTConfig) -> None:
    """Run only robot observation, WBC policy evaluation, and command output."""
    run_g1_control_loop(configure_ttt_config(config))


if __name__ == "__main__":
    main(tyro.cli(TTTConfig))
