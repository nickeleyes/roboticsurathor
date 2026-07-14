import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
MODEL_DATA = Path(__file__).resolve().parents[2] / "robot_model/model_data"
TABLE_XML = MODEL_DATA / "tictactoe_table.xml"
G1_XML = MODEL_DATA / "g1/g1_29dof_with_hand.xml"
HEAD_CAMERA_SITE = "head_camera"


def xml_vec(element, attr):
    return np.fromstring(element.attrib[attr], sep=" ")


def board_corners_world_from_xml():
    table = ET.parse(TABLE_XML).getroot().find(".//body[@name='tictactoe_table']")
    cells = [g for g in table.findall("geom") if g.attrib["name"].startswith("cell_")]
    pos, size = [np.array([xml_vec(cell, attr) for cell in cells]) for attr in ("pos", "size")]
    low = np.min(pos[:, :2] - size[:, :2], axis=0)
    high = np.max(pos[:, :2] + size[:, :2], axis=0)
    z = np.max(pos[:, 2] + size[:, 2])
    corners = np.array([
        [high[0], high[1], z],
        [high[0], low[1], z],
        [low[0], high[1], z],
        [low[0], low[1], z],
    ])
    return xml_vec(table, "pos") + corners

def default_pelvis_world_pos():
    pelvis = ET.parse(G1_XML).getroot().find(".//body[@name='pelvis']")
    return xml_vec(pelvis, "pos")


def head_camera_pose_in_pelvis():
    site = ET.parse(G1_XML).getroot().find(f".//site[@name='{HEAD_CAMERA_SITE}']")
    if site is None:
        raise ValueError(f"Could not find site {HEAD_CAMERA_SITE!r} in {G1_XML}")
    return xml_vec(site, "pos"), R.from_euler("xyz", xml_vec(site, "euler")).as_matrix()


def reconstruct_corners_from_depth(corners):
    pos, rot = head_camera_pose_in_pelvis()
    corners_camera = (rot.T @ (corners - pos).T).T
    depths = np.linalg.norm(corners_camera, axis=1)
    rays = corners_camera / depths[:, None]
    return pos + (rot @ (rays * depths[:, None]).T).T, depths

def cells_from_corners(corners):
    front_left, front_right, back_left, back_right = corners
    def board_point(row, col):
        r, c = (row + 0.5) / 3.0, (col + 0.5) / 3.0
        left = front_left + r * (back_left - front_left)
        right = front_right + r * (back_right - front_right)
        return left + c * (right - left)
    return {
        str(row * 3 + col + 1): board_point(row, col)
        for row in range(3)
        for col in range(3)
    }


def palm_down_rotation_from_corners(corners, pelvis_up):
    """Return a palm rotation aligned with the board plane.

    The palm frame's +Y axis points away from the palm, so it is aligned with
    the negative board normal. +X points from the back row toward the front row.
    """
    front_left, front_right, back_left, _ = corners
    toward_front = front_left - back_left
    toward_right = front_right - front_left
    toward_front /= np.linalg.norm(toward_front)
    toward_right /= np.linalg.norm(toward_right)

    normal = np.cross(toward_front, toward_right)
    normal /= np.linalg.norm(normal)
    if np.dot(normal, pelvis_up) < 0.0:
        normal = -normal

    palm_y = -normal
    palm_x = toward_front - np.dot(toward_front, palm_y) * palm_y
    palm_x /= np.linalg.norm(palm_x)
    palm_z = np.cross(palm_x, palm_y)
    palm_z /= np.linalg.norm(palm_z)
    return np.column_stack((palm_x, palm_y, palm_z))


def board_geometry_from_camera_depth(pelvis_pos=None, pelvis_rot=None):
    pelvis_pos = default_pelvis_world_pos() if pelvis_pos is None else pelvis_pos
    pelvis_rot = np.eye(3) if pelvis_rot is None else pelvis_rot
    corners_pelvis = (pelvis_rot.T @ (board_corners_world_from_xml() - pelvis_pos).T).T
    corners, depths = reconstruct_corners_from_depth(corners_pelvis)
    cells = cells_from_corners(corners)
    pelvis_up = pelvis_rot.T @ np.array([0.0, 0.0, 1.0])
    palm_rotation = palm_down_rotation_from_corners(corners, pelvis_up)
    print(
        "board from head RGBD:"
        f" corner depths {np.round(depths, 3)}"
        f" center pelvis {np.round(np.mean(list(cells.values()), axis=0), 3)}",
        flush=True,
    )
    return cells, palm_rotation


def board_cells_from_camera_depth(pelvis_pos=None, pelvis_rot=None):
    cells, _ = board_geometry_from_camera_depth(pelvis_pos, pelvis_rot)
    return cells
