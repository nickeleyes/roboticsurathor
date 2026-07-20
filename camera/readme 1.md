## Network Setup

The Unitree G1 uses a local 192.168.123.x network. Two main approaches for camera access:

| Device | IP | Role |
|--------|----|----|
| Jetson Orin NX (PC2) | 192.168.123.164 | ROS master, camera publisher |
| External PC | 192.168.123.51 | ROS subscriber, data consumer |

---

## Method 1 — ROS Noetic + RealSense D435i (`ros_noetic_connection.py`)

**Overview:** Stream Intel RealSense D435i camera (connected to Jetson Orin NX) over ROS Noetic to an external PC. Saves multiple streams: RGB, depth, aligned depth, IR, point cloud.

### Prerequisites

**On Jetson Orin NX (PC2):**
- ROS Noetic installed
- RealSense D435i connected via USB 3.0
- `realsense2_camera` ROS package installed

**On External PC:**
- ROS Noetic installed (or Python ROS bindings)
- Python 3.6+ with: `rospkg`, `catkin-pkg`, `opencv-python`, `numpy`

### Step 1 — Verify Camera Hardware (Jetson Only)

```bash
lsusb | grep -i intel
# Should show: Bus 002 Device 003: ID 8086:0b3a Intel Corp.

# Check USB 3.0 connection (must be 5000 Mbit/s, not 480)
for dev in /sys/bus/usb/devices/*/; do
    vendor=$(cat "$dev/idVendor" 2>/dev/null)
    if [ "$vendor" = "8086" ]; then
        echo "Speed: $(cat $dev/speed) Mbit/s"
    fi
done
```

**Troubleshooting:**
- **Speed: 480 Mbit/s** → USB 2.0 port. Move cable to a **blue USB 3.0 port**. Depth will not work on USB 2.0.
- **No Intel device** → Cable not seated. Try: `sudo udevadm control --reload-rules && sudo udevadm trigger`

### Step 2 — Set Environment Variables

**On Jetson (PC2):**
```bash
echo "source /opt/ros/noetic/setup.bash"                        >> ~/.bashrc
echo "export ROS_IP=192.168.123.164"                            >> ~/.bashrc
echo "export ROS_MASTER_URI=http://192.168.123.164:11311"       >> ~/.bashrc
source ~/.bashrc
```

**On External PC:**
```bash
echo "source /opt/ros/noetic/setup.bash"                        >> ~/.bashrc
echo "export ROS_IP=192.168.123.51"                             >> ~/.bashrc
echo "export ROS_MASTER_URI=http://192.168.123.164:11311"       >> ~/.bashrc
source ~/.bashrc
```

### Step 3 — Start ROS Master (Jetson Only)

Keep this terminal open at all times while streaming:
```bash
roscore &
sleep 5
```

**Expected output:**
```
process[master]: started with pid [XXXX]
ROS_MASTER_URI=http://ubuntu:11311/
started core service [/rosout]
```

**Troubleshooting:** If you get "Address already in use", kill stale roscore:
```bash
sudo pkill -f roscore && sleep 3
```

### Step 4 — Launch RealSense Node (Jetson Only)

Open a second terminal on the Jetson. This must stay open during streaming.

**Default (all streams enabled):**
```bash
roslaunch realsense2_camera rs_camera.launch \
  enable_color:=true          \
  enable_depth:=true          \
  enable_infra1:=true         \
  enable_infra2:=true         \
  enable_accel:=false         \
  enable_gyro:=false          \
  enable_pointcloud:=true     \
  align_depth:=true           \
  color_width:=640  color_height:=480  color_fps:=15 \
  depth_width:=640  depth_height:=480  depth_fps:=15
```

⚠️ **Important:** Always keep `enable_accel:=false` and `enable_gyro:=false`. The IMU interface causes USB control errors on Jetson Orin that interfere with depth streaming.

**Optional — Disable individual streams:**

| Stream | Launch Flag | SAVE_CONFIG Key |
|--------|-------------|-----------------|
| RGB color | `enable_color:=true/false` | `"color"` |
| Raw depth | `enable_depth:=true/false` | `"depth"` |
| Aligned depth | `align_depth:=true/false` | `"aligned"` |
| Left IR | `enable_infra1:=true/false` | `"infra1"` |
| Right IR | `enable_infra2:=true/false` | `"infra2"` |
| Point cloud | `enable_pointcloud:=true/false` | `"points"` |

**Example — Color and aligned depth only:**
```bash
roslaunch realsense2_camera rs_camera.launch \
  enable_color:=true   enable_depth:=true   \
  enable_infra1:=false enable_infra2:=false \
  enable_accel:=false  enable_gyro:=false   \
  enable_pointcloud:=false align_depth:=true \
  color_width:=640 color_height:=480 color_fps:=15 \
  depth_width:=640 depth_height:=480 depth_fps:=15
```

**Resolution/FPS options (reduce if depth is unstable):**

| Parameter | Options | Recommended |
|-----------|---------|-------------|
| `color_width / color_height` | 1280x720, 640x480, 424x240 | 640x480 |
| `color_fps` | 30, 15, 6 | 15 |
| `depth_width / depth_height` | 1280x720, 640x480, 424x240 | 640x480 |
| `depth_fps` | 30, 15, 6 | 15 |

**Expected output:**
```
[ INFO] setupStreams...
[ INFO] SELECTED BASE:Depth, 0
[ INFO] RealSense Node Is Up!
```

**Troubleshooting:**
- **"failed to set power state"** or **"uvc streamer watchdog triggered"** → Another process holds the camera. Run emergency reset (see below).
- **Depth publishes 0 Hz** → librealsense 2.50.0 is too old. Lower fps to 6 and resolution to 424x240, or upgrade librealsense.
- **Node crashes on start** → Check logs: `cat ~/.ros/log/latest/*.log | tail -30`
- Ignore `control_transfer WARNING` lines about index 768 — these are harmless.

### Step 5 — Verify Topics (Jetson Only)

```bash
rostopic list | grep camera
rostopic hz /camera/color/image_raw       --window 10
rostopic hz /camera/depth/image_rect_raw  --window 10
```

**Expected:** ~15 Hz for both topics.

### Step 6 — Connect from External PC

```bash
# Verify connection to ROS master
rostopic list | grep camera

# Check frame rates
rostopic hz /camera/color/image_raw
rostopic hz /camera/aligned_depth_to_color/image_raw
```

**Troubleshooting:**
- **"Unable to contact ROS master"** → Check `ping 192.168.123.164` and verify `ROS_MASTER_URI` is set.
- **Topics visible but no data** → `ROS_IP` is wrong on one or both machines. Must be numeric IP, not hostname.

### Step 7 — Run Subscriber Script (External PC)

**Install dependencies (once):**
```bash
pip install rospkg catkin-pkg opencv-python numpy
```

**Configure the script:**

Edit `SAVE_CONFIG` in `onboard/ros_noetic_connection.py` to match what the robot is publishing:

```python
SAVE_CONFIG = {
    "color":   True,   # /camera/color/image_raw
    "aligned": True,   # /camera/aligned_depth_to_color/image_raw
    "depth":   False,  # /camera/depth/image_rect_raw
    "infra1":  False,  # /camera/infra1/image_rect_raw
    "infra2":  False,  # /camera/infra2/image_rect_raw
    "points":  False,  # /camera/depth/color/points
}
```

⚠️ **Only enable streams that are actually launched on the robot** (Step 4). If you enable a stream here but disabled it in the launch command, the script will timeout waiting for it.

**Run:**
```bash
python3 onboard/ros_noetic_connection.py
```

**Expected output:**
```
📷 Connecting to robot camera topics...

Enabled topics:
  ✅ color
  ✅ aligned
  ❌ depth
  ❌ infra1
  ❌ infra2
  ❌ points

✅ Color RGB     → ./camera_captures/color_rgb.png  shape=(480, 640, 3)
✅ Aligned depth → ./camera_captures/aligned_depth_to_color.png
✅ IR left       → ./camera_captures/infrared_left.png  shape=(480, 640)

🎉 All frames saved!
📁 /home/user/projects/unitree_g1_perception/camera_captures/
   aligned_depth_to_color.png  (150.2 KB)
   color_rgb.png  (200.5 KB)
   infrared_left.png  (100.1 KB)
```

**Troubleshooting:**
- **Script times out** → That stream is not publishing. Check Step 5 and confirm the launch flags are `true`.
- **"No module named rospkg"** → Run `pip install rospkg catkin-pkg`.
- **"libp11-kit undefined symbol"** → Conda/system library conflict. Deactivate conda: `conda deactivate` (the script uses numpy decoder, not cv_bridge).

### Emergency Reset

Run this on PC2 whenever the camera stops responding or streams drop to 0 Hz:

```bash
sudo pkill -f roslaunch
sudo pkill -f nodelet
sleep 3

# Reset USB bus
echo -1 | sudo tee /sys/bus/usb/devices/usb2/authorized
sleep 2
echo  1 | sudo tee /sys/bus/usb/devices/usb2/authorized
sleep 3

# Confirm camera is back
lsusb | grep Intel
```

Then repeat Steps 3–4.

---

## Method 2 — Onboard UDP Camera (`onboard/udp_connection.py`)

**Purpose:** Receive the G1 head-camera H.264 RTP stream over UDP multicast and display frames with OpenCV.

**Requirements**

- Python: `ffmpeg-python`, `numpy`, `opencv-python`
- **ffmpeg** installed on the machine and on `PATH` (the script uses ffmpeg to demux/decode RTP → raw BGR)
- Network path to the robot’s camera multicast (same LAN / routing as the robot)

**Network / addresses** (edit constants at the top of `udp_connection.py` if yours differ)

| Constant     | Default           | Role |
|-------------|-------------------|------|
| `ROBOT_IP`  | `192.168.123.164` | Robot host IP (SDP `o=` line; should match the robot you are using) |
| `MCAST_IP`  | `230.1.1.1`       | Multicast group for the RTP stream |
| `MCAST_PORT`| `1720`            | UDP port for RTP video |

**Stream parameters (fixed in script):** 1280×720, H.264 RTP (`packetization-mode=1`), payload type 96.

**Run:** from repo root, `python cameras/onboard/udp_connection.py` (or your equivalent path). Quit the window with **Q**.
