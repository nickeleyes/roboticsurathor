# ROS 2 Camera Setup: Foxy Robot to Humble Thor

## Final architecture

```text
RealSense D435i
  -> robot computer 192.168.123.164
     Ubuntu 20.04, ROS 2 Foxy, camera publisher
  -> ROS 2 DDS network
  -> Jetson Thor 192.168.123.222
     existing GR00T container, Ubuntu 22.04, ROS 2 Humble
  -> camera/ros2_connection.py
  -> PNG images
```

Only the existing GR00T container is required on Thor. The temporary Noetic client-container approach was abandoned.

## What was installed

The robot already had ROS Noetic, ROS 2 Foxy, and the Noetic RealSense driver. It did not have the Foxy RealSense driver.

The robot cannot access the internet, so the following ARM64 Foxy packages were downloaded on Thor from the permanent ROS Foxy snapshot:

```text
ros-foxy-librealsense2 2.51.1
ros-foxy-realsense2-camera-msgs 4.51.1
ros-foxy-realsense2-camera 4.51.1
ros-foxy-realsense2-description 4.51.1
ros-foxy-xacro 2.0.7
```

The packages were checksum-verified, copied to `.164` with `scp`, simulated with APT, and installed locally. The simulation and installation showed five new packages, zero upgrades, and zero removals. Noetic and NVIDIA software were left installed.

## Why a clean Foxy shell is required

The robot's SSH login displays:

```text
ros:foxy(1) noetic(2) ?
```

This menu did not reliably isolate Foxy; `ROS_DISTRO` could still report Noetic. Mixing Foxy and Noetic paths in one shell is unsafe.

After SSH login, use:

```bash
env -i \
  HOME="$HOME" \
  USER="$USER" \
  TERM="$TERM" \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash --noprofile --norc

source /opt/ros/foxy/setup.bash
export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Verify that `echo "$ROS_DISTRO"` prints `foxy`.

## ROS 2 network configuration

Both `.164` and Thor use:

```bash
export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

ROS 2 does not use `roscore`, `ROS_MASTER_URI`, or `ROS_IP`. Foxy and Humble discover each other through DDS.

Foxy-to-Humble communication is not officially guaranteed, but it was verified on this hardware with a test string and actual camera images.

## Camera publisher

The current recommended Foxy launch on `.164` is:

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  enable_infra1:=false \
  enable_infra2:=false \
  enable_accel:=false \
  enable_gyro:=false \
  pointcloud.enable:=false \
  align_depth.enable:=true \
  rgb_camera.profile:=424,240,15 \
  depth_module.profile:=424,240,15 \
  initial_reset:=false
```

The D435i was verified as USB 3.2 with firmware `05.13.00.55`. Successful startup reports `RealSense Node Is Up!`.

## Clock correction

The robot initially reported a date in 1970, causing invalid camera timestamps. Its clock was manually synchronized to Thor before restarting the camera. A valid received timestamp was then confirmed.

This manual correction may not survive a robot reboot. Check the clocks if timestamps become incorrect again.

## Humble subscriber

The original `camera/ros_neotic_connection.py` uses ROS 1 `rospy` and cannot directly subscribe to ROS 2 topics.

The new subscriber is:

```text
camera/ros2_connection.py
```

It uses `rclpy` and ROS 2 sensor-data QoS. It captures enabled streams sequentially to reduce simultaneous large-message traffic. Each stream has its own 60-second timeout.

Configure saved streams in its `SAVE_CONFIG` dictionary:

```python
SAVE_CONFIG = {
    "color": True,
    "aligned": True,
    "depth": False,
    "infra1": False,
    "infra2": False,
}
```

From the GR00T repository root, run:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
python3 camera/ros2_connection.py
```

Current output is written to:

```text
/root/Projects/gr00t_wbc/camera_captures/
```

This differs from the older `camera/camera_captures/` directory. Files created by the root container may appear as `nobody:nogroup` on the host due to container UID mapping.

## Verified result

The working system saved:

```text
color_rgb.png
aligned_depth_to_color_raw.png
aligned_depth_to_color_colorized.png
```

A successful reduced-resolution capture reported color shape `(240, 424, 3)`. The saved color image was opened and visually verified.

## Why streams sometimes become slow or stale

Restarting the Foxy publisher on `.164` was observed to restore fast delivery. Restarting it resets several things at once:

- RealSense USB/UVC streams
- Foxy RealSense processing and alignment
- Foxy DDS participant and topic endpoints

Possible causes include:

1. The older RealSense 2.51.1 stack does not always recover from USB disturbances.
2. RGB and depth use different camera streams; one may continue while the other stalls.
3. Moving the robot may disturb the D435i USB cable or connector.
4. Each short-lived Humble subscriber must rediscover and rematch Foxy DDS endpoints.
5. Foxy-to-Humble DDS communication is unsupported and may reconnect inconsistently.
6. Large image messages require DDS/UDP fragmentation. Reducing resolution improved reliability.
7. Thor has several network interfaces, which can complicate DDS interface selection.

To distinguish camera failure from DDS failure, test locally on `.164` while the publisher is running:

```bash
ros2 topic echo /camera/color/image_raw --field header --once
ros2 topic echo /camera/depth/image_rect_raw --field header --once
```

- If a topic also fails locally, investigate the RealSense/USB pipeline.
- If it works locally but fails on Thor, investigate DDS/network matching.
- Seeing a topic in `ros2 topic list` does not prove that image data is flowing.

## Recovery

If capture becomes slow or times out:

1. Stop the Thor subscriber with `Ctrl+C`.
2. Stop the Foxy camera launch on `.164` with `Ctrl+C`.
3. Enter a clean Foxy shell again.
4. Set the three ROS environment variables.
5. Relaunch the camera and wait for `RealSense Node Is Up!`.
6. Run the Humble subscriber again.

Restart only the camera publisher when possible; a full robot or Thor reboot is normally unnecessary.

## Related files

- `camera/readme 1.md`: original Noetic design
- `camera/readme 3.md`: short run-only procedure
- `camera/ros_neotic_connection.py`: legacy ROS 1 subscriber
- `camera/ros2_connection.py`: current ROS 2 subscriber

