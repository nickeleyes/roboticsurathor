import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2

# --- Config ---
SAVE_DIR = "./camera_captures"
os.makedirs(SAVE_DIR, exist_ok=True)

# Mandatory: color RGB and aligned depth are always saved
# Optional: set to True to save, False to skip
SAVE_CONFIG = {
    "color":   True,     # Mandatory
    "aligned": True,     # Mandatory
    "depth":   False,    # Optional
    "infra1":  False,    # Optional
    "infra2":  False,    # Optional
    "points":  False,    # Optional
}

saved = {
    "color":   False,
    "depth":   False,
    "aligned": False,
    "infra1":  False,
    "infra2":  False,
    "points":  False,
}

def all_saved():
    # Check mandatory items (color, aligned) and enabled optional items
    for key in SAVE_CONFIG:
        if SAVE_CONFIG[key] and not saved[key]:
            return False
    return True

# ── Manual image decoding (no cv_bridge) ──────────────

def ros_img_to_numpy(msg):
    """Convert ROS Image message to numpy array without cv_bridge."""
    dtype_map = {
        "rgb8":   (np.uint8,  3),
        "bgr8":   (np.uint8,  3),
        "rgba8":  (np.uint8,  4),
        "mono8":  (np.uint8,  1),
        "mono16": (np.uint16, 1),
        "16UC1":  (np.uint16, 1),
        "32FC1":  (np.float32,1),
    }
    encoding = msg.encoding
    if encoding not in dtype_map:
        # fallback — try uint8
        dtype, channels = np.uint8, 1
    else:
        dtype, channels = dtype_map[encoding]

    arr = np.frombuffer(msg.data, dtype=dtype)

    if channels == 1:
        arr = arr.reshape(msg.height, msg.width)
    else:
        arr = arr.reshape(msg.height, msg.width, channels)

    # Convert RGB to BGR for OpenCV
    if encoding == "rgb8":
        arr = arr[:, :, ::-1]

    return arr

# ── Callbacks ──────────────────────────────────────────

def color_callback(msg):
    if saved["color"]:
        return
    frame = ros_img_to_numpy(msg)
    path = f"{SAVE_DIR}/color_rgb.png"
    cv2.imwrite(path, frame)
    saved["color"] = True
    print(f"✅ Color RGB     → {path}  shape={frame.shape}")

def depth_callback(msg):
    if saved["depth"]:
        return
    frame = ros_img_to_numpy(msg)
    # Save raw 16-bit depth
    cv2.imwrite(f"{SAVE_DIR}/depth_raw.png", frame)
    # Save colorized for visualization
    depth_vis = cv2.convertScaleAbs(frame, alpha=0.03)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite(f"{SAVE_DIR}/depth_colorized.png", depth_colored)
    saved["depth"] = True
    print(f"✅ Depth raw     → {SAVE_DIR}/depth_raw.png  shape={frame.shape}")
    print(f"✅ Depth color   → {SAVE_DIR}/depth_colorized.png")

def aligned_callback(msg):
    if saved["aligned"]:
        return
    frame = ros_img_to_numpy(msg)
    depth_vis = cv2.convertScaleAbs(frame, alpha=0.03)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite(f"{SAVE_DIR}/aligned_depth_to_color.png", depth_colored)
    saved["aligned"] = True
    print(f"✅ Aligned depth → {SAVE_DIR}/aligned_depth_to_color.png")

def infra1_callback(msg):
    if saved["infra1"]:
        return
    frame = ros_img_to_numpy(msg)
    path = f"{SAVE_DIR}/infrared_left.png"
    cv2.imwrite(path, frame)
    saved["infra1"] = True
    print(f"✅ IR left       → {path}  shape={frame.shape}")

def infra2_callback(msg):
    if saved["infra2"]:
        return
    frame = ros_img_to_numpy(msg)
    path = f"{SAVE_DIR}/infrared_right.png"
    cv2.imwrite(path, frame)
    saved["infra2"] = True
    print(f"✅ IR right      → {path}  shape={frame.shape}")

def pointcloud_callback(msg):
    if saved["points"]:
        return
    points = list(pc2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True
    ))
    arr = np.array(points, dtype=np.float32)
    path = f"{SAVE_DIR}/pointcloud.npy"
    np.save(path, arr)
    saved["points"] = True
    print(f"✅ Point cloud   → {path}  points={len(arr)}")

# ── Main ───────────────────────────────────────────────

def main():
    rospy.init_node("save_camera_frames", anonymous=True)

    print("\n📷 Connecting to robot camera topics...\n")
    print("Enabled topics:")
    for key, enabled in SAVE_CONFIG.items():
        status = "✅" if enabled else "❌"
        print(f"  {status} {key}")
    print()

    # Subscribe only to enabled topics
    if SAVE_CONFIG["color"]:
        rospy.Subscriber("/camera/color/image_raw",
                         Image, color_callback)
    if SAVE_CONFIG["depth"]:
        rospy.Subscriber("/camera/depth/image_rect_raw",
                         Image, depth_callback)
    if SAVE_CONFIG["aligned"]:
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw",
                         Image, aligned_callback)
    if SAVE_CONFIG["infra1"]:
        rospy.Subscriber("/camera/infra1/image_rect_raw",
                         Image, infra1_callback)
    if SAVE_CONFIG["infra2"]:
        rospy.Subscriber("/camera/infra2/image_rect_raw",
                         Image, infra2_callback)
    if SAVE_CONFIG["points"]:
        rospy.Subscriber("/camera/depth/color/points",
                         PointCloud2, pointcloud_callback)

    rate = rospy.Rate(10)
    timeout = rospy.Time.now() + rospy.Duration(30)

    while not rospy.is_shutdown():
        if all_saved():
            print("\n🎉 All frames saved!")
            print(f"📁 {os.path.abspath(SAVE_DIR)}/")
            for f in sorted(os.listdir(SAVE_DIR)):
                size = os.path.getsize(f"{SAVE_DIR}/{f}")
                print(f"   {f}  ({size/1024:.1f} KB)")
            break

        if rospy.Time.now() > timeout:
            print("\n⚠️  Timeout! Results:")
            for k, v in saved.items():
                print(f"   {k}: {'✅' if v else '❌ not received'}")
            break

        pending = [k for k, v in saved.items() if not v]
        print(f"⏳ Waiting for: {pending}    ", end="\r")
        rate.sleep()

if __name__ == "__main__":
    main()