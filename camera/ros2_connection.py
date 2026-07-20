import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


SAVE_DIR = "./camera_captures"
TIMEOUT_SECONDS = 30.0

SAVE_CONFIG = {
    "color": True,
    "aligned": True,
    "depth": False,
    "infra1": False,
    "infra2": False,
}

TOPICS = {
    "color": "/camera/color/image_raw",
    "aligned": "/camera/aligned_depth_to_color/image_raw",
    "depth": "/camera/depth/image_rect_raw",
    "infra1": "/camera/infra1/image_rect_raw",
    "infra2": "/camera/infra2/image_rect_raw",
}


def ros_img_to_numpy(msg):
    dtype_map = {
        "rgb8": (np.uint8, 3),
        "bgr8": (np.uint8, 3),
        "rgba8": (np.uint8, 4),
        "mono8": (np.uint8, 1),
        "mono16": (np.uint16, 1),
        "16UC1": (np.uint16, 1),
        "32FC1": (np.float32, 1),
    }
    dtype, channels = dtype_map.get(msg.encoding, (np.uint8, 1))
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    item_size = dtype.itemsize
    row_values = msg.step // item_size
    data = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_values)
    useful_values = msg.width * channels
    data = data[:, :useful_values]

    if channels == 1:
        frame = data.reshape(msg.height, msg.width)
    else:
        frame = data.reshape(msg.height, msg.width, channels)

    if msg.encoding == "rgb8":
        frame = frame[:, :, ::-1]
    elif msg.encoding == "rgba8":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
    return frame


class CameraFrameSaver(Node):
    def __init__(self):
        super().__init__("save_camera_frames")
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.saved = {key: False for key in SAVE_CONFIG}
        self.started_at = time.monotonic()
        self.finished = False
        self._camera_subscriptions = []

        callbacks = {
            "color": self.color_callback,
            "aligned": self.aligned_callback,
            "depth": self.depth_callback,
            "infra1": self.infra1_callback,
            "infra2": self.infra2_callback,
        }

        print("\nConnecting to ROS 2 camera topics:\n")
        for key, enabled in SAVE_CONFIG.items():
            print(f"  {'enabled ' if enabled else 'disabled'} {key}: {TOPICS[key]}")
            if enabled:
                self._camera_subscriptions.append(
                    self.create_subscription(
                        Image,
                        TOPICS[key],
                        callbacks[key],
                        qos_profile_sensor_data,
                    )
                )
        print()
        self.timer = self.create_timer(0.1, self.check_finished)

    def all_saved(self):
        return all(not enabled or self.saved[key] for key, enabled in SAVE_CONFIG.items())

    def save_depth_images(self, frame, prefix):
        cv2.imwrite(f"{SAVE_DIR}/{prefix}_raw.png", frame)
        depth_vis = cv2.convertScaleAbs(frame, alpha=0.03)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(f"{SAVE_DIR}/{prefix}_colorized.png", depth_colored)

    def color_callback(self, msg):
        if self.saved["color"]:
            return
        frame = ros_img_to_numpy(msg)
        path = f"{SAVE_DIR}/color_rgb.png"
        cv2.imwrite(path, frame)
        self.saved["color"] = True
        print(f"Saved color: {path}, shape={frame.shape}")

    def depth_callback(self, msg):
        if self.saved["depth"]:
            return
        frame = ros_img_to_numpy(msg)
        self.save_depth_images(frame, "depth")
        self.saved["depth"] = True
        print(f"Saved raw and colorized depth in {SAVE_DIR}")

    def aligned_callback(self, msg):
        if self.saved["aligned"]:
            return
        frame = ros_img_to_numpy(msg)
        self.save_depth_images(frame, "aligned_depth_to_color")
        self.saved["aligned"] = True
        print(f"Saved aligned depth in {SAVE_DIR}")

    def infra1_callback(self, msg):
        if self.saved["infra1"]:
            return
        frame = ros_img_to_numpy(msg)
        path = f"{SAVE_DIR}/infrared_left.png"
        cv2.imwrite(path, frame)
        self.saved["infra1"] = True
        print(f"Saved left infrared: {path}, shape={frame.shape}")

    def infra2_callback(self, msg):
        if self.saved["infra2"]:
            return
        frame = ros_img_to_numpy(msg)
        path = f"{SAVE_DIR}/infrared_right.png"
        cv2.imwrite(path, frame)
        self.saved["infra2"] = True
        print(f"Saved right infrared: {path}, shape={frame.shape}")

    def check_finished(self):
        if self.all_saved():
            print(f"\nAll requested frames saved in {os.path.abspath(SAVE_DIR)}")
            for filename in sorted(os.listdir(SAVE_DIR)):
                path = os.path.join(SAVE_DIR, filename)
                if os.path.isfile(path):
                    print(f"  {filename} ({os.path.getsize(path) / 1024:.1f} KB)")
            self.finished = True
            self.timer.cancel()
        elif time.monotonic() - self.started_at > TIMEOUT_SECONDS:
            print("\nTimed out waiting for camera frames:")
            for key, enabled in SAVE_CONFIG.items():
                if enabled:
                    print(f"  {key}: {'saved' if self.saved[key] else 'not received'}")
            self.finished = True
            self.timer.cancel()


def main():
    rclpy.init()
    node = CameraFrameSaver()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
