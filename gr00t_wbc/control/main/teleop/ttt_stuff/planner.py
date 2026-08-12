import numpy as np

HEIGHTS = {"high": 0.14, "close": 0.10, "grab": 0.025}


def unit(v):
    return v / np.linalg.norm(v)


def bilerp(p1, p3, p7, p9, u, v):
    return (1 - u) * (1 - v) * p1 + u * (1 - v) * p3 + (1 - u) * v * p7 + u * v * p9


class BoardPlanner:
    def points(self, corners):
        p1, p3, p7, p9 = np.array(corners, dtype=float)
        points = {
            f"pos{1 + 3 * r + c}": bilerp(p1, p3, p7, p9, c / 2, r / 2)
            for r in range(3)
            for c in range(3)
        }
        x_step = (
            (points["pos3"] - points["pos1"])
            + (points["pos9"] - points["pos7"])
        ) / 4
        down_step = (
            (points["pos7"] - points["pos1"])
            + (points["pos9"] - points["pos3"])
        ) / 4
        x_axis = unit(x_step)
        normal = unit(np.cross(x_axis, down_step))
        if normal[2] < 0:
            normal = -normal
        rotation = np.column_stack((x_axis, -normal, unit(np.cross(normal, x_axis))))
        for i in range(5):
            v = -0.1 + 0.3 * i
            # Vision normalizes the portrait paper back to its original landscape
            # frame. Canonical left/right therefore map to physical bottom/top.
            points[f"green_square{i + 1}"] = bilerp(p1, p3, p7, p9, -0.4, v)
            points[f"white_square{i + 1}"] = bilerp(p1, p3, p7, p9, 1.4, v)

        return points, rotation

    def plan_steps(self, board, move):
        color = "green" if move[0].lower() == "g" else "white"
        value = "1" if color == "green" else "2"
        piece = f"{color}_square{board.count(value) + 1}"
        goal = f"pos{move[1]}"

        # Each tuple is (target name, target height, grabbing). The middle finger
        # closes while descending onto the source piece, stays closed during
        # transport, and opens while placing the piece in the destination cell.
        return [
            # Verify every depot and board position before handling a piece.
            # ("white_square1", 0.06, True),
            # ("white_square2", 0.06, True),
            # ("white_square3", 0.06, True),
            # ("white_square4", 0.06, True),
            # ("white_square5", 0.06, True),
            # ("pos1", 0.06, True),
            # ("pos2", 0.06, True),
            # ("pos3", 0.06, True),
            # ("pos4", 0.06, True),
            # ("pos5", 0.06, True),
            # ("pos6", 0.06, True),
            # ("pos7", 0.06, True),
            # ("pos8", 0.06, True),
            # ("pos9", 0.06, True),
            # ("green_square1", 0.06, True),
            # ("green_square2", 0.06, True),
            # ("green_square3", 0.06, True),
            # ("green_square4", 0.06, True),
            # ("green_square5", 0.06, True),

            # Typical move.
            ("pos5", "high", False),
            (piece, "close", False),
            (piece, "grab", False),
            (piece, "grab", True),
            (piece, "close", True),
            (goal, "close", True),
            (goal, "grab", True),
            (goal, "grab", False),
            (goal, "close", False),
            ("pos5", "high", False),
        ]
    #- rotation[:, 0] * 0.015  - rotation[:, 2] * 0.010
    def target(self, corners, name, height):
        points, rotation = self.points(corners)
        height = HEIGHTS[height] if isinstance(height, str) else height
        normal = -rotation[:, 1]
        position = points[name] + rotation[:, 2] * 0.01
        if name.startswith("pos"):
            position += (rotation[:, 2] - rotation[:, 0]) * 0.02
        angle = np.deg2rad(15.0)
        rotation = rotation @ np.array([[np.cos(angle), -np.sin(angle), 0],
                                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        return position + normal * height, rotation
