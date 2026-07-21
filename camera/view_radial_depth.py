"""Interactively inspect a saved radial-distance capture.

This viewer is completely separate from camera capture. It opens a window only
when this file is run explicitly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
CAPTURE_DIR = REPO_ROOT / "camera_captures"
COLOR_PATH = CAPTURE_DIR / "color_rgb.png"
RADIAL_PATH = CAPTURE_DIR / "radial_distance_m.npy"


def main():
    if not COLOR_PATH.exists() or not RADIAL_PATH.exists():
        raise FileNotFoundError(
            "Missing capture files. Expected:\n"
            f"  {COLOR_PATH}\n"
            f"  {RADIAL_PATH}\n"
            "Run camera/ros2_connection.py first."
        )

    color = np.asarray(Image.open(COLOR_PATH).convert("RGB"))
    radial_m = np.load(RADIAL_PATH)
    if color.shape[:2] != radial_m.shape:
        raise ValueError(
            f"RGB shape {color.shape[:2]} does not match radial map {radial_m.shape}"
        )

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.imshow(color)
    axis.set_title("Move the cursor over the image to inspect radial distance")
    status = axis.text(
        0.01,
        0.01,
        "Pixel: --   Radial distance: --",
        transform=axis.transAxes,
        color="white",
        fontsize=11,
        bbox={"facecolor": "black", "alpha": 0.75, "pad": 4},
    )

    def on_move(event):
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return
        u = int(round(event.xdata))
        v = int(round(event.ydata))
        if not (0 <= u < radial_m.shape[1] and 0 <= v < radial_m.shape[0]):
            return
        distance = float(radial_m[v, u])
        value = f"{distance:.4f} m" if np.isfinite(distance) else "invalid"
        status.set_text(f"Pixel: ({u}, {v})   Radial distance: {value}")
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("motion_notify_event", on_move)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
