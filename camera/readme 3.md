# Run the ROS 2 Camera Connection

This starts the existing setup:

- Robot camera computer: `192.168.123.164`, ROS 2 Foxy publisher
- Jetson Thor: `192.168.123.222`, existing GR00T ROS 2 Humble container
- ROS domain: `23`

The Foxy RealSense packages and the Humble subscriber are already installed.

## Terminal 1 — Start the camera publisher on the robot

From a fresh terminal on Thor, connect to the robot camera computer:

```bash
ssh unitree@192.168.123.164
```

If the login displays this selection prompt:

```text
ros:foxy(1) noetic(2) ?
```

enter `1` to continue. The menu may still leave `ROS_DISTRO` set to Noetic, so do not rely on the environment it selects.

Open a clean shell so the installed Noetic and Foxy environments are not mixed:

```bash
env -i \
  HOME="$HOME" \
  USER="$USER" \
  TERM="$TERM" \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash --noprofile --norc
```

Load and configure ROS 2 Foxy:

```bash
source /opt/ros/foxy/setup.bash
export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Confirm that only Foxy is active:

```bash
echo "$ROS_DISTRO"
which ros2
which rostopic || echo "Good: Noetic is absent"
```

The expected ROS distribution is `foxy`, and `rostopic` should be absent.

Start the RealSense D435i publisher:

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  enable_infra1:=false \
  enable_infra2:=false \
  enable_accel:=false \
  enable_gyro:=false \
  enable_sync:=true \
  pointcloud.enable:=false \
  align_depth.enable:=true \
  rgb_camera.profile:=424,240,15 \
  depth_module.profile:=424,240,15 \
  initial_reset:=false
```

Wait until the terminal reports:

```text
RealSense Node Is Up!
```

Keep this terminal open.

## Terminal 2 — Enter the existing GR00T container on Thor

From a fresh terminal on Thor, make sure the existing container is running:

```bash
sudo docker start gr00t_wbc-bash-root
```

Enter it:

```bash
sudo docker exec -it gr00t_wbc-bash-root bash
```

Go to the repository:

```bash
cd /root/Projects/gr00t_wbc
```

Load and configure ROS 2 Humble:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Run the camera subscriber:

```bash
python3 camera/ros2_connection.py
```xhost +si:localuser:root

The requested frames are saved under:

```text
/root/Projects/gr00t_wbc/camera_captures/
```

This capture contains `color_rgb.png`, `radial_distance_m.npy`,
`radial_distance_visualization.png`, and `camera_intrinsics.json`.

## Run live camera vision into the simulation

On the Thor host, temporarily authorize the container's root user to use X11:

```bash
xhost +si:localuser:root
```

Then, inside the GR00T container:

```bash
export DISPLAY=:1
python3 gr00t_wbc/control/main/teleop/run_ttt.py --interface sim
```

Press the Logitech F310 `X` button. It captures synchronized RGB and depth,
detects the board, localizes it, plans the move, and moves only the simulated robot.
The RealSense publisher in Terminal 1 must still be running.

The `xhost` permission is temporary and normally ends with the current graphical login session.

## Stop

The subscriber exits after saving the requested frames. To stop the camera publisher, press `Ctrl+C` in Terminal 1.
