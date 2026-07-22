"""Minimal Linux reader for the Logitech F310 gamepad."""

import struct
import threading
from queue import Empty, SimpleQueue


class GamepadController:
    EVENT = struct.Struct("llHHi")
    AXES = {0: "x", 1: "y", 3: "rx"}  # ABS_X, ABS_Y, ABS_RX
    BUTTONS = {
        315: "]",
        304: "9",
        310: "camera_x",  # LB
        311: "camera_z",  # RB
        307: "program",
    }

    def __init__(self):
        self.path = "/dev/input/event7"
        self.file = open(self.path, "rb", buffering=0)
        self.deadzone = 0.08
        self.response_exponent = 3.0
        self.max_linear = .7
        self.max_angular = .7
        self.axes = {"x": 0.0, "y": 0.0, "rx": 0.0}
        self.pressed_keys = SimpleQueue()
        threading.Thread(target=self._read, daemon=True).start()
        print(f"Controller: Logitech Gamepad F310 ({self.path})")

    def _read(self):
        try:
            while data := self.file.read(self.EVENT.size):
                _, _, event_type, code, value = self.EVENT.unpack(data)
                if event_type == 3 and code in self.AXES:  # EV_ABS
                    value = max(-1.0, min(1.0, value / 32768.0))
                    self.axes[self.AXES[code]] = self._shape_axis(value)
                elif event_type == 1 and value == 1:
                    if code in self.BUTTONS:
                        self.pressed_keys.put(self.BUTTONS[code])
        except (OSError, ValueError) as error:
            print(f"Gamepad reader stopped: {error}")
            self.axes = {"x": 0.0, "y": 0.0, "rx": 0.0}

    def _shape_axis(self, value):
        magnitude = abs(value)
        if magnitude <= self.deadzone:
            return 0.0

        normalized = (magnitude - self.deadzone) / (1.0 - self.deadzone)
        shaped = normalized**self.response_exponent
        return shaped if value > 0.0 else -shaped

    def get_velocity(self):
        return [
            -self.axes["y"] * self.max_linear,
            -self.axes["x"] * self.max_linear,
            -self.axes["rx"] * self.max_angular,
        ]

    def get_key(self):
        try:
            return self.pressed_keys.get_nowait()
        except Empty:
            return None

    def close(self):
        self.file.close()
