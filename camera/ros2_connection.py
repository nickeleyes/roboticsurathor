"""Save one synchronized RGB/depth pair, intrinsics, and radial-distance map."""
import json
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = Path("camera_captures")
TIMEOUT = 60
MAX_SYNC_NS = 50_000_000  # 50 ms


def timestamp_ns(message):
    stamp = message.header.stamp
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class CameraCapture(Node):
    def __init__(self):
        super().__init__("synchronized_camera_capture")
        OUTPUT.mkdir(exist_ok=True)
        self.color = self.depth = self.info = None
        self.started = time.monotonic()
        self.done = False
        topics = {
            "color": "/camera/color/image_raw",
            "depth": "/camera/aligned_depth_to_color/image_raw",
        }
        for name, topic in topics.items():
            self.create_subscription(
                Image, topic, lambda msg, key=name: self.receive(key, msg),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            CameraInfo, "/camera/color/camera_info", self.receive_info,
            qos_profile_sensor_data,
        )
        self.create_timer(0.1, self.check_timeout)
        print("Waiting for synchronized RGB, aligned depth, and camera info...")

    def receive(self, name, message):
        setattr(self, name, message)
        self.try_capture()

    def receive_info(self, message):
        self.info = message
        self.try_capture()

    def try_capture(self):
        if self.done or any(item is None for item in (self.color, self.depth, self.info)):
            return
        difference = abs(timestamp_ns(self.color) - timestamp_ns(self.depth))
        if difference > MAX_SYNC_NS:
            return
        self.save_capture(difference)
        self.done = True

    def save_capture(self, difference):
        if self.color.encoding != "rgb8" or self.depth.encoding != "16UC1":
            raise ValueError(
                f"Expected rgb8/16UC1; got {self.color.encoding}/{self.depth.encoding}"
            )
        rgb = np.frombuffer(self.color.data, np.uint8).reshape(
            self.color.height, self.color.width, 3
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth = np.frombuffer(self.depth.data, np.uint16).reshape(
            self.depth.height, self.depth.width
        )
        z = depth.astype(np.float32) / 1000.0

        fx, cx = self.info.k[0], self.info.k[2]
        fy, cy = self.info.k[4], self.info.k[5]
        u, v = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
        radial = np.sqrt(((u - cx) * z / fx) ** 2 +
                         ((v - cy) * z / fy) ** 2 + z ** 2).astype(np.float32)
        radial[z == 0] = np.nan

        cv2.imwrite(str(OUTPUT / "color_rgb.png"), bgr)
        np.save(OUTPUT / "radial_distance_m.npy", radial)
        intrinsics = {
            "width": self.info.width,
            "height": self.info.height,
            "frame_id": self.info.header.frame_id,
            "distortion_model": self.info.distortion_model,
            "distortion_coefficients": list(self.info.d),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }
        (OUTPUT / "camera_intrinsics.json").write_text(
            json.dumps(intrinsics, indent=2), encoding="utf-8"
        )

        figure, axis = plt.subplots()
        image = axis.imshow(radial, cmap="turbo")
        axis.set_title("Radial distance")
        figure.colorbar(image, ax=axis, label="meters")
        figure.savefig(OUTPUT / "radial_distance_visualization.png", dpi=150)
        plt.close(figure)
        print(f"Saved synchronized capture ({difference / 1e6:.3f} ms difference)")
        print(f"Output: {OUTPUT.resolve()}")

    def check_timeout(self):
        if not self.done and time.monotonic() - self.started > TIMEOUT:
            print("Timed out waiting for synchronized camera data")
            self.done = True


def main():
    rclpy.init()
    node = CameraCapture()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
