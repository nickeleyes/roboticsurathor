import numpy as np

HEIGHTS = {"high": 0.10, "close": 0.06, "grab": 0.00}


def unit(v):
    return v / np.linalg.norm(v)


def bilerp(p1, p3, p7, p9, u, v):
    return (1 - u) * (1 - v) * p1 + u * (1 - v) * p3 + (1 - u) * v * p7 + u * v * p9


class BoardPlanner:
    def points(self, corners):
        p1, p3, p7, p9 = np.array(corners, dtype=float)
        z = np.mean([p1[2], p3[2], p7[2], p9[2]])
        for point in (p1, p3, p7, p9):
            point[2] = z
        points = {f"pos{1 + 3 * r + c}": bilerp(p1, p3, p7, p9, c / 2, r / 2) for r in range(3) for c in range(3)}
        x_step = ((points["pos3"] - points["pos1"]) + (points["pos9"] - points["pos7"])) / 4
        x_axis = unit(np.array([x_step[0], x_step[1], 0.0]))
        normal = np.array([0.0, 0.0, 1.0])
        rotation = np.column_stack((x_axis, -normal, unit(np.cross(normal, x_axis))))
        rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]) @ rotation
        for i in range(5):
            u = -0.1 + 0.3 * i
            points[f"green_square{i + 1}"] = bilerp(p1, p3, p7, p9, u, 1.4)
            points[f"white_square{i + 1}"] = bilerp(p1, p3, p7, p9, u, -0.4)
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

    def target(self, corners, name, height):
        points, rotation = self.points(corners)
        return points[name] + [0.0, 0.0, HEIGHTS[height]], rotation
