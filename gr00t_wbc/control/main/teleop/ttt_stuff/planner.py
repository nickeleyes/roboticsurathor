import numpy as np

HEIGHTS = {"high": 0.10, "close": 0.06, "grab": 0.02}


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
        for i in range(5):
            v = -0.1 + 0.3 * i
            points[f"green_square{i + 1}"] = bilerp(p1, p3, p7, p9, -0.4, v)
            points[f"white_square{i + 1}"] = bilerp(p1, p3, p7, p9, 1.4, v)
        return points, rotation

    def plan_steps(self, board, move):
        color = "green" if move[0].lower() == "g" else "white"
        value = "1" if color == "green" else "2"
        piece = f"{color}_square{board.count(value) + 1}"
        goal = f"pos{move[1]}"

        return [
            ("pos5", "high"),
            (piece, "close"),
            (piece, "grab"),
            (piece, "close"),
            (goal, "close"),
            (goal, "grab"),
            (goal, "close"),
            ("pos5", "high"),
        ]

    def target(self, corners, name, height):
        points, rotation = self.points(corners)
        return points[name] + [0.0, 0.0, HEIGHTS[height]], rotation
