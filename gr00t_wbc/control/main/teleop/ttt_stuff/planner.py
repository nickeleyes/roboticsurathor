import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

MODEL_DATA = Path(__file__).resolve().parents[3] / "robot_model/model_data"
TABLE_XML = MODEL_DATA / "tictactoe_table.xml"
G1_XML = MODEL_DATA / "g1/g1_29dof_with_hand.xml"

HEIGHTS = {"high": 0.21, "close": 0.18, "grab": 0.07}


def unit(v):
    return v / np.linalg.norm(v)


def bilerp(p1, p3, p7, p9, u, v):
    return (1 - u) * (1 - v) * p1 + u * (1 - v) * p3 + (1 - u) * v * p7 + u * v * p9


def xml_vec(element, attr):
    return np.fromstring(element.attrib[attr], sep=" ")


def board_corners_from_xml():
    table = ET.parse(TABLE_XML).getroot().find(".//body[@name='tictactoe_table']")
    cells = {g.attrib["name"]: g for g in table.findall("geom")}
    corners = []
    for name in ("cell_1", "cell_3", "cell_7", "cell_9"):
        cell = cells[name]
        point = xml_vec(cell, "pos")
        point[2] += xml_vec(cell, "size")[2]
        corners.append(point)
    return xml_vec(table, "pos") + corners


def corners_in_pelvis(corners, floating_base_pose=None):
    if floating_base_pose is None:
        pelvis = ET.parse(G1_XML).getroot().find(".//body[@name='pelvis']")
        pelvis_pos = xml_vec(pelvis, "pos")
        pelvis_rot = np.eye(3)
    else:
        pose = np.array(floating_base_pose, dtype=float)
        pelvis_pos = pose[:3]
        pelvis_rot = R.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return (pelvis_rot.T @ (corners - pelvis_pos).T).T


class BoardPlanner:
    def points(self, corners):
        p1, p3, p7, p9 = np.array(corners, dtype=float)
        points = {
            f"pos{1 + 3 * r + c}": bilerp(p1, p3, p7, p9, c / 2, r / 2)
            for r in range(3)
            for c in range(3)
        }
        x_step = ((points["pos3"] - points["pos1"]) + (points["pos9"] - points["pos7"])) / 4
        normal = unit(np.cross(p1 - p7, p3 - p1))
        if normal[2] < 0.0:
            normal = -normal
        x_axis = unit((p1 - p7) - np.dot(p1 - p7, normal) * normal)
        rotation = np.column_stack((x_axis, -normal, unit(np.cross(normal, x_axis))))
        for i in range(5):
            v = -0.1 + 0.3 * i
            points[f"green_square{i + 1}"] = bilerp(p1, p3, p7, p9, -0.4, v)
            points[f"white_square{i + 1}"] = bilerp(p1, p3, p7, p9, 1.4, v)
        return points, normal, rotation

    def plan_move(self, board, move, corners):
        points, normal, rotation = self.points(corners)
        color = "green" if move[0].lower() == "g" else "white"
        value = "1" if color == "green" else "2"
        piece = f"{color}_square{board.count(value) + 1}"
        goal = f"pos{move[1]}"
        points[piece] += np.array([0.0, 0.0, 0.1])
        points[goal] += np.array([0.0, 0.0, 0.1])
        steps = [
            ("pos5", "high"),
            (piece, "close"), (piece, "grab"),
            (piece, "close"),
            (goal, "close"), (goal, "grab"), (goal, "close"),
            ("pos5", "high"),
        ]
        return [step if step[0] == "gripper" else self.pose(points, normal, rotation, *step) for step in steps]

    def pose(self, points, normal, rotation, name, height):
        return "move", points[name] + normal * HEIGHTS[height], rotation, height
