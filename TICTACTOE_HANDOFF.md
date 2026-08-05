# G1 Tic-Tac-Toe Handoff

This is an experimental G1 whole-body-control application that detects a physical
tic-tac-toe board, chooses a minimax move, and guides the right hand through the
pick-and-place waypoints. It supports the MuJoCo simulator and the real robot.

> **Do not copy only `ttt_stuff/`.** The application depends on changes throughout
> this GR00T-WBC repository. Share or clone the complete repository.

## Getting the complete project

This project is based on NVIDIA's GR00T Whole-Body Control repository and is shared
through the `nickeleyes/TTTG1` repository.

```bash
git lfs install
git clone https://github.com/nickeleyes/TTTG1.git
cd TTTG1
git lfs pull
```

Git LFS is required. In particular,
`gr00t_wbc/control/main/teleop/ttt_stuff/center_resnet18.pt` is a roughly 43 MB
ResNet checkpoint. If that file is a tiny text file containing an LFS pointer,
run `git lfs pull` before starting the program. A GitHub source ZIP is not the
recommended way to obtain this repository.

For the base system requirements and Docker environment, follow [README.md](README.md).
This work was developed in the repository's GR00T Docker environment, which supplies
the ROS 2, MuJoCo, PyTorch, Pinocchio/Pink, and G1 control dependencies.

## Quick start in simulation

Run these commands from the repository root, inside the project container:

```bash
python gr00t_wbc/control/main/teleop/run_ttt.py --interface sim --offline
```

`--offline` reads the checked-in RGB image, radial-depth array, and camera
intrinsics from `camera_captures/`. This is the most reproducible first run because
it does not require a live RealSense publisher.

The simulator uses the tic-tac-toe table in
`gr00t_wbc/control/robot_model/model_data/tictactoe_table.xml`. The current default
model also displays a red marker at the active IK target and locks the simulated
feet when a move begins.

### Keyboard flow

Focus the terminal running `run_ttt.py` when using these controls.

1. Press `]` to activate the lower-body balance policy.
2. Wait for the three-step arm initialization to finish. The terminal will print
   `Arm initialization complete`.
3. Press `p` to capture/process the board and prepare a move.
4. Press `SPACE` when prompted to begin, and again to accept each of the eight
   waypoints. Each waypoint runs live IK until it is accepted.
5. The backtick key (`` ` ``) triggers this process's keyboard emergency stop.
   Keep the robot's physical emergency stop available as the primary real-robot
   safeguard.

Do not advance a waypoint just because the prompt appears. Check the hand target,
joint motion, and printed IK error first.

`o` disables the lower-body policy, but it does **not** cancel the active TTT arm
thread and therefore must not be treated as the arm emergency stop.

## Running with the real camera and robot

The real-robot path should only be used by someone familiar with the G1 safety and
network procedures. Clear the workspace, keep an emergency stop available, begin
with reduced-risk tests, and never leave the robot unattended.

The camera path expects synchronized ROS 2 messages on these exact topics:

| Topic | Required message/encoding |
| --- | --- |
| `/camera/color/image_raw` | `sensor_msgs/Image`, `rgb8` |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image`, `16UC1` depth in millimeters |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` |

The capture code accepts color/depth timestamps no more than 50 ms apart and times
out after 60 seconds. The tested RealSense launch and ROS domain/network setup are
recorded in [`camera/readme 3.md`](camera/readme%203.md). That older note contains
historical controls; for the current program use `p` and `SPACE` as described above.

After the camera topics are visible in the same ROS domain, run:

```bash
python gr00t_wbc/control/main/teleop/run_ttt.py --interface real
```

The real-robot implementation assumes both feet remain planted. It estimates pelvis
drift from left- and right-foot forward kinematics so the board target remains fixed
relative to the world while the torso/waist moves.

## What the program does

The runtime data flow is:

1. `run_ttt.py` configures waist control, right-arm gravity compensation, a 10
   degree/second upper-body speed limit, and the shared G1 control loop.
2. `ros2capture.py` receives a synchronized RGB/depth/intrinsics sample, or
   `program.py` loads the saved sample in offline mode.
3. `vision.py` finds the white paper, perspective-rectifies the grid, and uses the
   checked-in ResNet-18 checkpoint to classify all nine cells.
4. `localization.py` combines pixel locations with radial depth and the calibrated
   camera-to-pelvis transform to obtain the four corner-cell centers in meters.
5. `engine.py` chooses the next move using minimax.
6. `planner.py` creates eight approach, grab, carry, place, and retreat waypoints.
7. `controller.py` continuously solves right-arm plus waist-yaw IK. `program.py`
   publishes the resulting upper-body targets to the main WBC loop.

Board strings contain nine row-major characters:

| Code | Meaning |
| --- | --- |
| `0` | Empty cell |
| `1` | Green piece |
| `2` | White piece |

Moves use `g1` through `g9` or `w1` through `w9`, where cell 1 is the rectified
top-left and cell 9 is the bottom-right.

## Why `ttt_stuff/` alone does not run

The most important dependencies outside that directory are:

| Path | Why it is needed |
| --- | --- |
| `gr00t_wbc/control/main/teleop/run_ttt.py` | Executable entry point and TTT-specific WBC configuration |
| `gr00t_wbc/control/main/teleop/run_g1_control_loop.py` | Startup, `p`/`SPACE` callbacks, TTT simulation messages, and command keepalive integration |
| `gr00t_wbc/control/envs/g1/g1_env.py` | Right-arm gravity compensation and simulation hooks |
| `gr00t_wbc/control/envs/g1/sim/base_sim.py` | Simulated foot locks and IK target marker |
| `gr00t_wbc/control/policy/g1_decoupled_whole_body_policy.py` | Preserves waist-yaw commands from the upper-body IK path |
| `gr00t_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml` | Tuned G1 arm gains used by this project |
| `gr00t_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf` | Adds the calibrated `right_hand_grip_center` IK frame |
| `gr00t_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.xml` | Simulation grip target, camera/table inclusion, foot welds, and marker bodies |
| `gr00t_wbc/control/robot_model/model_data/tictactoe_table.xml` | Simulator board/table geometry |
| `gr00t_wbc/control/robot_model/supplemental_info/g1/g1_supplemental_info.py` | Adds the `waist_yaw_only` joint group used by the IK solver |
| `camera_captures/` | Reproducible input for `--offline` |

There are additional repository changes for the Jetson/ARM64 Docker setup, camera
experiments, gamepad work, and ONNX Runtime provider selection. Because the project
evolved in one working repository, a hand-selected folder copy or partial cherry-pick is
not a supported handoff. Start from the complete repository, then separate components
only after a successful baseline run.

## Calibration and tuning assumptions

Several values are experiment-specific and are currently hard-coded:

- The camera extrinsics in `localization.py` use a pitch of about 0.831 radians and
  a camera position of `[0.056, 0.0, 0.504]` meters in the pelvis frame.
- Vision rectifies the paper to 1000 by 774 pixels and crops a 690 by 690 grid.
- The localization geometry assumes the tested US Letter board layout and a
  roughly 128 mm span between the outer cell centers.
- Planner offsets are 100 mm (`high`), 60 mm (`close`), and 20 mm (`grab`) above
  the localized board plane.
- Five green pieces are expected along the left side and five white pieces along
  the right side of the board, at the positions encoded in `planner.py`.
- The grip frame and fixed finger pose are tuned for the tested hand and pieces.

A different camera mount, printed board, table height, piece layout, or gripper will
require recalibration. Test perception and target coordinates before permitting
physical motion.

## Troubleshooting

### The checkpoint cannot be loaded

Run `git lfs pull`, then check that `center_resnet18.pt` is tens of megabytes rather
than a short text pointer. Also confirm that `torch` and `torchvision` import in the
same environment used to launch the control loop.

### `No board-paper corners detected`

The detector looks for a large bright quadrilateral that does not touch the image
edge. Check exposure, glare, paper contrast, occlusion, and whether the full paper is
inside the color image.

### `No valid depth near ...`

The aligned depth frame has too few finite positive samples near a required cell
center. Confirm depth-to-color alignment, remove reflective/occluding objects, and
check the `16UC1` encoding.

### Camera capture times out

Use `ros2 topic list` and `ros2 topic echo --once /camera/color/camera_info` in the
same container/environment. Check `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY`, middleware,
host networking, topic names, and that color/depth timestamps are synchronized.

### The detected board is wrong

Run with the saved capture first. The classifier was trained only for `empty`,
`green`, and `white`; inspect lighting and the rectified cell ordering before
changing planner or IK code.

### The hand target is consistently offset

Do not compensate by blindly changing waypoints. Verify, in order, the camera
extrinsics, radial-depth calculation, printed-board dimensions, pelvis-frame
transform, and `right_hand_grip_center` frame.

## Maintainer checklist before sharing a GitHub link

A push includes commits, not uncommitted working-tree files. From the repository
root:

```bash
git status --short
git diff --check
git diff --cached --stat
```

Stage only the intended source, model/configuration, capture, and documentation
files; review the staged diff; then commit. In particular, make sure the current
`ttt_stuff/program.py` is tracked. Add the GitHub repository as a remote (this
example uses the name `tttg1`), then push both normal Git objects and LFS objects:

```bash
git remote add tttg1 https://github.com/nickeleyes/TTTG1.git
git push -u tttg1 HEAD:main
git lfs push --all tttg1 HEAD
```

Finally, test a clean clone using the commands in **Getting the complete project**.
That clean-clone test is the reliable proof that the grad student will receive all
code and large assets.
