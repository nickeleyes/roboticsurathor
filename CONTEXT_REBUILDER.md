# GR00T Camera and Tic-Tac-Toe Context Rebuilder

Use this file to restore context in a new Codex chat. Ask Codex to read this file and inspect the listed files before making changes.

## Goal

Use the G1 head RealSense camera to locate a physical tic-tac-toe board, transform its four grid-corner centers into the pelvis frame, and use those points for right-arm inverse kinematics. Development is currently simulation-only.

## Machines and ROS

- Robot computer: `192.168.123.164`, Ubuntu 20.04, ROS 2 Foxy.
- Jetson Thor: `192.168.123.222`, GR00T Docker, Ubuntu 22.04, ROS 2 Humble.
- Foxy RealSense packages were installed offline on the robot.
- Both sides use ROS domain `23`, Fast DDS, and `ROS_LOCALHOST_ONLY=0`.
- When SSH asks `foxy(1) noetic(2)`, select Foxy (`1`).

## Camera Pipeline

1. Start the RealSense ROS 2 publisher on `.164` following `camera/readme 3.md`.
2. In the GR00T container on `.222`, run:

   ```bash
   python3 camera/ros2_connection.py
   ```

3. The synchronized capture writes the required RGB, radial depth, and camera intrinsics into `camera_captures/`.
4. `camera/view_radial_depth.py` provides the optional hover-based radial-distance check.
5. `camera/camerafeedtester.py` detects the board and classifies its cells from the saved RGB image.
6. `camera/localization.py` maps p1, p3, p7, and p9 through the saved depth and intrinsics, transforms them into the pelvis frame, regularizes the board geometry, and writes `camera_captures/board_localization.json`.

Canonical board orientation is green storage on the left and white storage on the right. No image or corner rotation should be added. Corner order is p1 top-left, p3 top-right, p7 bottom-left, p9 bottom-right.

## Important Localization Assumptions

- Camera-to-pelvis translation: `[0.056, 0.0, 0.504]` metres.
- The transform includes the D435 mounting pitch and optical-frame conversion.
- Localization is based on a picture taken while the torso initially faces forward.
- The saved JSON contains the board state and four grid-corner centers in the pelvis frame.
- The current saved batch is temporary test input; production should capture and localize at runtime.

## Simulation

Run from the repository root inside the GR00T container:

```bash
python3 gr00t_wbc/control/main/teleop/run_ttt.py --interface SIM
```

Press the controller's **X** button to start the planned move.

Current behavior:

1. Four large spheres display p1, p3, p7, and p9.
2. The waist turns left by `WAIST_YAW` in `run_ttt.py` (currently 20 degrees).
3. The pelvis may translate or rotate while balancing.
4. The original board location is retained in the simulator world frame.
5. Live pelvis pose is used to transform that fixed board back into the current pelvis frame.
6. The marker positions and IK targets are continuously corrected.
7. The waist returns to neutral after the move.

This is intentionally guarded as simulation-only. Do not remove that guard or test this path on the physical robot without a separate safety review.

## IK Design

Relevant file: `gr00t_wbc/control/main/teleop/ttt_stuff/controller.py`.

- The right palm's local green/Y axis is its normal.
- That palm normal is constrained to the board normal.
- Rotation about the palm normal remains free.
- Position cost is 40 and palm-normal cost is 50.
- Full palm orientation was rejected because it unnecessarily constrained twist and worsened reachability.
- Some locations remain difficult without waist rotation.

## Continuous-Correction Smoothing

Relevant file: `gr00t_wbc/control/main/teleop/run_ttt.py`.

The first live implementation jittered because it recomputed IK every 0.2 seconds from a noisy pelvis pose and repeatedly replaced short-horizon targets. It now uses:

- `CORNER_BLEND = 0.12` to filter corrected board corners.
- `JOINT_BLEND = 0.18` to interpolate toward new IK solutions.
- `COMMAND_HORIZON = 1.0` seconds.
- Four-second arm movement phases and 1.5-second gripper phases.

These are the first values to tune if motion remains too fast, slow, or jittery.

## MuJoCo Crash Already Fixed

Repeated marker updates initially caused a native segmentation fault. The cause was calling `mj_forward()` and changing the MuJoCo model from the control thread while the simulator thread was stepping it.

The fix in `gr00t_wbc/control/envs/g1/sim/base_sim.py` queues marker coordinates under a lock and applies them inside `sim_step()`. Do not restore cross-thread MuJoCo calls.

## Elastic Band

The simulator's original elastic-band behavior is enabled and must remain unchanged. Attempts to disable or modify it caused falling and severe jitter. It is important to simulation balance, but it also contributes to gradual pelvis motion after a waist turn; continuous pose compensation addresses that drift.

## Files to Inspect First

- `camera/readme 3.md`
- `camera/ros2_connection.py`
- `camera/camerafeedtester.py`
- `camera/localization.py`
- `camera_captures/board_localization.json`
- `gr00t_wbc/control/main/teleop/run_ttt.py`
- `gr00t_wbc/control/main/teleop/ttt_stuff/planner.py`
- `gr00t_wbc/control/main/teleop/ttt_stuff/controller.py`
- `gr00t_wbc/control/main/teleop/run_g1_control_loop.py`
- `gr00t_wbc/control/envs/g1/g1_env.py`
- `gr00t_wbc/control/envs/g1/sim/base_sim.py`
- `gr00t_wbc/control/policy/g1_decoupled_whole_body_policy.py`
- `gr00t_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.xml`

## Current Next Step

Restart the simulation fully, press X, and evaluate whether the filtered live correction makes the arm sufficiently slow and smooth. If it still jitters, inspect logged/current pelvis pose and successive IK targets before changing the elastic band. Tune the blend constants and command durations conservatively.

## New-Chat Prompt

```text
Read CONTEXT_REBUILDER.md completely, then inspect the files it lists and the current git diff. Continue from the documented current next step. Preserve the simulation-only guard, canonical board orientation, palm-normal-only IK constraint, original elastic-band behavior, and thread-safe MuJoCo marker updates.
```
