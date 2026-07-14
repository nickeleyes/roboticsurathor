from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from gr00t_wbc.control.main.teleop.movement.planner import Waypoint
from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.robot_model.robot_model import ReducedRobotModel
from gr00t_wbc.control.teleop.solver.body.body_ik_solver import BodyIKSolver
from gr00t_wbc.control.teleop.solver.body.body_ik_solver_settings import BodyIKSolverSettings


class UnreachableTarget(RuntimeError):
    pass


@dataclass(frozen=True)
class JointCommand:
    name: str
    upper_body_pose: np.ndarray
    duration: float


class MovementController:
    """Convert Cartesian waypoints into checked G1 upper-body commands."""

    OPEN_HAND = np.zeros(7)
    CLOSED_HAND = np.array([-0.5, -0.7, -0.7, 1.5, 1.5, 0.6, 1.5])

    def __init__(
        self,
        enable_waist: bool = False,
        position_tolerance: float = 0.025,
        orientation_tolerance_degrees: float = 8.0,
    ):
        waist_location = "lower_and_upper_body" if enable_waist else "lower_body"
        self.robot = instantiate_g1_robot_model(waist_location=waist_location)
        self.enable_waist = enable_waist
        self.frame = "right_hand_palm_center"
        self.robot.supplemental_info.hand_frame_names["right"] = self.frame

        active_groups = ["right_arm"]
        if enable_waist:
            active_groups.append("waist_pitch_only")
        self.body = ReducedRobotModel.from_active_groups(self.robot, active_groups)
        settings = BodyIKSolverSettings()
        settings.num_step_per_frame = 50
        self.solver = BodyIKSolver(settings)
        self.solver.register_robot(self.body)
        self.solver.update_weights(
            {self.frame: {"position_cost": 100.0, "orientation_cost": 25.0}}
        )

        self.position_tolerance = position_tolerance
        self.orientation_tolerance = np.deg2rad(orientation_tolerance_degrees)
        self.upper_indices = list(self.robot.get_joint_group_indices("upper_body"))
        self.arm_indices = self.robot.get_joint_group_indices("right_arm")
        self.hand_indices = self.robot.get_hand_actuated_joint_indices("right")
        self.arm_in_upper = [self.upper_indices.index(index) for index in self.arm_indices]
        self.hand_in_upper = [self.upper_indices.index(index) for index in self.hand_indices]
        self.waist_pitch_index = self.robot.dof_index("waist_pitch_joint")
        self.waist_pitch_in_upper = (
            self.upper_indices.index(self.waist_pitch_index) if enable_waist else None
        )

    def commands(self, plan: list[Waypoint], current_q) -> list[JointCommand]:
        """Solve the complete plan before returning any command."""
        predicted_q = np.asarray(current_q, dtype=float).copy()
        if predicted_q.shape != (self.robot.num_dofs,):
            raise ValueError(f"expected q shape {(self.robot.num_dofs,)}, got {predicted_q.shape}")

        commands = []
        for waypoint in plan:
            command, predicted_q = self._solve(waypoint, predicted_q)
            commands.append(command)
        return commands

    def _solve(self, waypoint: Waypoint, seed_q):
        self._seed(seed_q)
        target = np.eye(4)
        target[:3, :3] = waypoint.rotation
        target[:3, 3] = waypoint.position

        best = None
        for _ in range(4):
            solved_reduced = self.solver({self.frame: target})
            solved_full = self.body.reduced_to_full_configuration(solved_reduced)
            candidate_q = seed_q.copy()
            candidate_q[self.arm_indices] = solved_full[self.arm_indices]
            if self.enable_waist:
                candidate_q[self.waist_pitch_index] = solved_full[self.waist_pitch_index]
            errors = self._pose_error(candidate_q, target)
            score = errors[0] / self.position_tolerance + errors[1] / self.orientation_tolerance
            if best is None or score < best[0]:
                best = score, candidate_q, errors
            if errors[0] <= self.position_tolerance and errors[1] <= self.orientation_tolerance:
                break

        _, candidate_q, (position_error, orientation_error) = best
        if position_error > self.position_tolerance or orientation_error > self.orientation_tolerance:
            raise UnreachableTarget(
                f"{waypoint.name}: position error {position_error * 1000:.1f} mm, "
                f"orientation error {np.rad2deg(orientation_error):.1f} deg"
            )

        upper_body = seed_q[self.upper_indices].copy()
        upper_body[self.arm_in_upper] = candidate_q[self.arm_indices]
        upper_body[self.hand_in_upper] = self._hand_pose(waypoint.hand)
        if self.enable_waist:
            upper_body[self.waist_pitch_in_upper] = candidate_q[self.waist_pitch_index]
        candidate_q[self.hand_indices] = self._hand_pose(waypoint.hand)
        return (
            JointCommand(
                name=waypoint.name,
                upper_body_pose=upper_body,
                duration=waypoint.duration,
            ),
            candidate_q,
        )

    def _seed(self, full_q):
        reduced_q = self.body.full_to_reduced_configuration(full_q)
        reduced_q = self.body.clip_configuration(reduced_q, margin=1e-4)
        configuration = self.solver.configuration
        configuration.q = reduced_q
        configuration.update()
        self.body.cache_forward_kinematics(reduced_q)
        self.solver.tasks["posture"].set_target_from_configuration(configuration)

    def _pose_error(self, q, target):
        self.robot.cache_forward_kinematics(q)
        actual = self.robot.frame_placement(self.frame).homogeneous
        position = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        orientation = R.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude()
        return position, orientation

    def _hand_pose(self, state):
        if state == "open":
            return self.OPEN_HAND
        if state == "closed":
            return self.CLOSED_HAND
        raise ValueError(f"unknown hand state: {state}")
