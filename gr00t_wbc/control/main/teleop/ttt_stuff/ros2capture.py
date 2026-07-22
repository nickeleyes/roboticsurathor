"""Receive one synchronized RGB/depth pair from the ROS 2 RealSense publisher."""
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class CameraCapture(Node):
    def __init__(self):
        super().__init__("ttt_camera_capture")
        self.color = self.depth = self.info = self.result = None
        self.create_subscription(Image, "/camera/color/image_raw",
                                 lambda msg: self.receive("color", msg), qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw",
                                 lambda msg: self.receive("depth", msg), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/color/camera_info",
                                 lambda msg: self.receive("info", msg), qos_profile_sensor_data)

    def receive(self, name, message):
        setattr(self, name, message)
        if self.result is not None or any(x is None for x in (self.color, self.depth, self.info)):
            return
        color_ns = self.color.header.stamp.sec * 1_000_000_000 + self.color.header.stamp.nanosec
        depth_ns = self.depth.header.stamp.sec * 1_000_000_000 + self.depth.header.stamp.nanosec
        if abs(color_ns - depth_ns) > 50_000_000:
            return
        if self.color.encoding != "rgb8" or self.depth.encoding != "16UC1":
            raise ValueError(f"Expected rgb8/16UC1, got {self.color.encoding}/{self.depth.encoding}")
        rgb = np.frombuffer(self.color.data, np.uint8).reshape(self.color.height,
                                                               self.color.width, 3)
        z = np.frombuffer(self.depth.data, np.uint16).reshape(self.depth.height,
                                                              self.depth.width).astype(float) / 1000
        fx, fy, cx, cy = self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5]
        u, v = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
        radial = np.sqrt(((u - cx) * z / fx) ** 2 + ((v - cy) * z / fy) ** 2 + z ** 2)
        radial[z == 0] = np.nan
        self.result = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), radial, {
            "width": self.info.width, "height": self.info.height,
            "distortion_coefficients": list(self.info.d),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        }


def capture():
    node = CameraCapture()
    executor = SingleThreadedExecutor(context=node.context)
    executor.add_node(node)
    deadline = time.monotonic() + 60
    try:
        while rclpy.ok() and node.result is None and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()
    if node.result is None:
        raise TimeoutError("Timed out waiting for synchronized camera data")
    return node.result
