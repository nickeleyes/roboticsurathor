# Camera Setup Instructions: New Thor and New GR00T Container

This guide configures a fresh Jetson Thor/GR00T installation to receive Intel RealSense D435i images from the robot camera computer.

## Target system

| Machine | Address | ROS role |
|---|---:|---|
| Robot camera computer | `192.168.123.164` | ROS 2 Foxy camera publisher |
| Jetson Thor | `192.168.123.222` | ROS 2 Humble subscriber inside GR00T |

The camera is connected to `.164`. Thor uses only the normal GR00T container; do not create a separate Noetic container.

## 1. Prepare the new Thor host

Configure Thor's robot-network interface with:

```text
Address: 192.168.123.222
Subnet mask: 255.255.255.0
```

Verify the robot computer is reachable:

```bash
ping -c 3 192.168.123.164
```

Clone and install GR00T if it is not already present:

```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install

mkdir -p ~/Projects
cd ~/Projects
git clone <REPOSITORY_URL_CONTAINING_THIS_CAMERA_SETUP> gr00t_wbc
cd gr00t_wbc
git checkout <CAMERA_SETUP_BRANCH>
```

Use the repository/branch where `camera/ros2_connection.py` and this README have been committed. The unmodified upstream repository may not contain these camera additions.

Install/start the root GR00T container using the repository helper:

```bash
./docker/run_docker.sh --install --root
```

The helper creates the container with host networking and mounts the repository under `/root/Projects/gr00t_wbc`.

For later sessions, enter it with:

```bash
cd ~/Projects/gr00t_wbc
./docker/run_docker.sh --root
```

## 2. Verify the new GR00T container

Inside the container:

```bash
cat /etc/os-release
source /opt/ros/humble/setup.bash
echo "$ROS_DISTRO"
which ros2
python3 --version
```

Expected essentials:

```text
Ubuntu 22.04
ROS_DISTRO=humble
/opt/ros/humble/bin/ros2
Python 3.10
```

Verify the Python camera dependencies:

```bash
python3 -c "import rclpy; from sensor_msgs.msg import Image; import cv2, numpy; print('dependencies OK')"
```

If that succeeds, leave the container with:

```bash
exit
```

## 3. Verify Foxy RealSense on the robot computer

Connect to the robot:

```bash
ssh unitree@192.168.123.164
```

The login may display:

```text
ros:foxy(1) noetic(2) ?
```

Enter `1` to reach a shell, but do not trust the menu to isolate Foxy. Start a clean shell:

```bash
env -i \
  HOME="$HOME" \
  USER="$USER" \
  TERM="$TERM" \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash --noprofile --norc
```

Load only Foxy:

```bash
source /opt/ros/foxy/setup.bash
```

Verify the installed driver:

```bash
echo "$ROS_DISTRO"
ros2 pkg list | grep realsense
ros2 pkg executables realsense2_camera
```

The expected packages include:

```text
realsense2_camera
realsense2_camera_msgs
realsense2_description
```

If those packages appear, skip to Step 4.

### If the Foxy RealSense packages are missing

The robot has no internet route. Download the ARM64 packages on Thor from the permanent Foxy archive, then transfer them to `.164`.

On Thor:

```bash
FOXY_DEB_DIR=$(mktemp -d /tmp/foxy-realsense-debs.XXXXXX)
SNAPSHOT_BASE='http://snapshots.ros.org/foxy/final/ubuntu/pool/main/r'

curl -fL --output-dir "$FOXY_DEB_DIR" -O \
"$SNAPSHOT_BASE/ros-foxy-librealsense2/ros-foxy-librealsense2_2.51.1-1focal.20230527.034906_arm64.deb"

curl -fL --output-dir "$FOXY_DEB_DIR" -O \
"$SNAPSHOT_BASE/ros-foxy-realsense2-camera-msgs/ros-foxy-realsense2-camera-msgs_4.51.1-1focal.20230527.054107_arm64.deb"

curl -fL --output-dir "$FOXY_DEB_DIR" -O \
"$SNAPSHOT_BASE/ros-foxy-realsense2-camera/ros-foxy-realsense2-camera_4.51.1-1focal.20230606.035542_arm64.deb"

curl -fL --output-dir "$FOXY_DEB_DIR" -O \
"$SNAPSHOT_BASE/ros-foxy-xacro/ros-foxy-xacro_2.0.7-1focal.20230527.044107_arm64.deb"

curl -fL --output-dir "$FOXY_DEB_DIR" -O \
"$SNAPSHOT_BASE/ros-foxy-realsense2-description/ros-foxy-realsense2-description_4.51.1-1focal.20230527.083652_arm64.deb"
```

Verify that five files exist:

```bash
ls -lh "$FOXY_DEB_DIR"
md5sum "$FOXY_DEB_DIR"/*.deb
```

Expected hashes:

```text
46d70de52d33e5e5c27b96938a7dabf6  ros-foxy-librealsense2_2.51.1-1focal.20230527.034906_arm64.deb
a620595d0281e9fa1809f83bfc2a2820  ros-foxy-realsense2-camera-msgs_4.51.1-1focal.20230527.054107_arm64.deb
99b3852b82438cdca3330b84b7589838  ros-foxy-realsense2-camera_4.51.1-1focal.20230606.035542_arm64.deb
76e675b1abd3160c49faaf0c022fd079  ros-foxy-realsense2-description_4.51.1-1focal.20230527.083652_arm64.deb
fa7d117736ef9f1b436a52a3b5c94fbe  ros-foxy-xacro_2.0.7-1focal.20230527.044107_arm64.deb
```

Create a destination on `.164`:

```bash
ssh unitree@192.168.123.164 'mkdir -p /tmp/foxy-realsense-install && chmod 755 /tmp/foxy-realsense-install'
```

Transfer the files:

```bash
scp "$FOXY_DEB_DIR"/*.deb unitree@192.168.123.164:/tmp/foxy-realsense-install/
```

On `.164`, verify and simulate before installing:

```bash
chmod 644 /tmp/foxy-realsense-install/*.deb
md5sum /tmp/foxy-realsense-install/*.deb
sudo apt-get -s install /tmp/foxy-realsense-install/*.deb
```

The simulation must report five new packages and zero removals. Then install:

```bash
sudo apt-get install /tmp/foxy-realsense-install/*.deb
```

Do not run `apt autoremove`.

## 4. Check the clocks

On Thor:

```bash
date -Ins
```

On `.164`:

```bash
date -Ins
```

The times must be close. If `.164` reports 1970, synchronize it before starting the camera. Obtain Thor's Unix time:

```bash
date +%s
```

On `.164`, insert that value:

```bash
sudo date -s '@THOR_TIMESTAMP'
```

## 5. Start the Foxy camera publisher on `.164`

In the clean Foxy shell:

```bash
source /opt/ros/foxy/setup.bash
export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Verify:

```bash
echo "$ROS_DISTRO"
echo "$ROS_DOMAIN_ID"
echo "$ROS_LOCALHOST_ONLY"
echo "$RMW_IMPLEMENTATION"
```

Launch the camera:

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

Wait for:

```text
RealSense Node Is Up!
```

Keep this terminal open.

## 6. Start the Humble subscriber on Thor

Open a second Thor terminal and enter the GR00T container:

```bash
cd ~/Projects/gr00t_wbc
./docker/run_docker.sh --root
```

Inside the container:

```bash
cd /root/Projects/gr00t_wbc
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Verify discovery:

```bash
ping -c 2 192.168.123.164
ros2 topic list | grep '^/camera'
```

Run the subscriber already included in this repository:

```bash
python3 camera/ros2_connection.py
```

Successful output includes:

```text
Saved color
Saved aligned depth
All requested frames saved
```

The current output directory is:

```text
/root/Projects/gr00t_wbc/camera_captures/
```

Expected files:

```text
color_rgb.png
aligned_depth_to_color_raw.png
aligned_depth_to_color_colorized.png
```

## 7. Stop safely

The subscriber exits after saving the requested images. Stop the Foxy publisher by pressing `Ctrl+C` in its `.164` terminal.

Closing the terminals does not uninstall packages or delete the Docker container.

## Troubleshooting

### Foxy shell still reports Noetic

Do not source Foxy on top of Noetic. Run the clean `env -i` shell from Step 3, then source Foxy again.

### Topics appear but images time out

Test locally on `.164`:

```bash
ros2 topic echo /camera/color/image_raw --field header --once
ros2 topic echo /camera/depth/image_rect_raw --field header --once
```

- Local failure indicates a RealSense/USB stream problem.
- Local success with Thor failure indicates DDS/network matching.

### Publisher becomes slow or stale

Stop it with `Ctrl+C`, re-enter the clean Foxy environment, and relaunch it. This resets the USB streams, alignment process, and Foxy DDS endpoints.

### Color works but aligned depth does not

Inspect the `.164` launch output for `Frames didn't arrive`, watchdog, or USB errors. Check the D435i USB cable and test `/camera/depth/image_rect_raw` locally.

### Files cannot be deleted from the host

The root container can create files with container-mapped ownership. This is a Docker UID/ownership issue, not a camera failure. Avoid recursively changing unrelated project ownership.

## Notes

- Foxy-to-Humble communication works here but is not officially guaranteed by ROS 2.
- The original Noetic packages remain installed on `.164`.
- Do not run Noetic and Foxy camera publishers simultaneously; only one process can own the D435i.
- For daily startup after setup is complete, use `camera/readme 3.md`.
