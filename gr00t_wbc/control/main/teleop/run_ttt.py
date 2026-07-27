import threading
import time

import numpy as np

import tyro

from gr00t_wbc.control.main.constants import CONTROL_GOAL_TOPIC, STATE_TOPIC_NAME
from gr00t_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from gr00t_wbc.control.main.teleop.run_g1_control_loop import main as run_g1_control_loop
from gr00t_wbc.control.main.teleop.ttt_stuff.controller import Controller
from gr00t_wbc.control.main.teleop.ttt_stuff.engine import move_code
from gr00t_wbc.control.main.teleop.ttt_stuff.localization import localize
from gr00t_wbc.control.main.teleop.ttt_stuff.pelvis_tracking import PelvisTracker
from gr00t_wbc.control.main.teleop.ttt_stuff.planner import (
    BoardPlanner,
)
from gr00t_wbc.control.main.teleop.ttt_stuff.ros2capture import capture
from gr00t_wbc.control.main.teleop.ttt_stuff.vision import detect
from gr00t_wbc.control.utils.ros_utils import ROSMsgPublisher, ROSMsgSubscriber

POSE_DURATION = 5.0
POSE_BLEND = 0.18
SETTLE_DURATION = 0.5
YAW_KEY_COUNT = 4
YAW_KEY_PERIOD = 1.0


def main(config: ControlLoopConfig):
    config.enable_waist = True
    planner = BoardPlanner()
    controller = Controller(0.0)
    pelvis_tracker = PelvisTracker(translation_tolerance=0.06)
    publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
    move_thread = None
    policy_key_event = 0

    def send(joints, duration, corners=None, preserve_waist=False):
        goal = {
            "target_upper_body_pose": joints,
            "target_time": time.monotonic() + duration,
            "preserve_upper_body_waist_yaw": preserve_waist,
        }
        if corners is not None:
            goal["ttt_corners"] = corners.tolist()
        publisher.publish(goal)

    def press_policy_key(key, count):
        nonlocal policy_key_event
        for _ in range(count):
            policy_key_event += 1
            publisher.publish({
                "policy_key": key,
                "policy_key_event": policy_key_event,
            })
            time.sleep(YAW_KEY_PERIOD)

    def run_move():
        nonlocal controller
        frame, radial, intrinsics = capture()
        data = localize(detect(frame), radial, intrinsics)
        board = data["board_state"]
        corners = np.array([
            data["corner_cell_centers"][name] for name in ("p1", "p3", "p7", "p9")
        ])
        state = state_subscriber.get_msg()
        if state is None or state.get("q") is None:
            print("Robot joint state unavailable; move cancelled", flush=True)
            return
        starting_joints = np.asarray(state["q"])[controller.upper_body].copy()
        publisher.publish({"ttt_corners": corners.tolist()})
        move = move_code(board)
        if move is None:
            print("No legal move available.")
            return

        pelvis_tracker.capture_initial(np.asarray(state["q"], dtype=float))
        print("Sending GR00T yaw key 8 four times", flush=True)
        press_policy_key("8", YAW_KEY_COUNT)
        time.sleep(POSE_DURATION)

        state = state_subscriber.get_msg()
        if state is None or state.get("q") is None:
            print("Final leg joint state unavailable; move cancelled", flush=True)
            return
        waist_index = controller.model.dof_index("waist_yaw_joint")
        achieved_waist_yaw = float(np.asarray(state["q"])[waist_index])
        print(
            f"GR00T rotation settled with waist at "
            f"{np.degrees(achieved_waist_yaw):.2f} degrees",
            flush=True,
        )

        # Seed arm IK from the pose actually achieved by GR00T.
        controller = Controller(achieved_waist_yaw)
        smooth_joints = controller.side_t_pose(achieved_waist_yaw)
        time.sleep(SETTLE_DURATION)

        state = state_subscriber.get_msg()
        if state is None or state.get("q") is None:
            print("Settled leg joint state unavailable; move cancelled", flush=True)
            return
        try:
            pelvis_tracker.capture_final(np.asarray(state["q"], dtype=float))
            tracking = pelvis_tracker.pelvis_delta()
            corners = pelvis_tracker.transform_points(corners)
        except (RuntimeError, ValueError) as error:
            print(f"Pelvis tracking failed; move cancelled: {error}", flush=True)
            return
        delta = tracking["final_pelvis_T_initial_pelvis"]
        rotation_degrees = np.degrees(
            np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
        )
        print(
            "Pelvis correction from fixed-foot FK: "
            f"translation={np.round(delta[:3, 3], 4)} m, "
            f"rotation={rotation_degrees:.2f} deg, "
            f"foot disagreement={tracking['translation_disagreement_m']:.4f} m/"
            f"{np.degrees(tracking['rotation_disagreement_rad']):.2f} deg",
            flush=True,
        )
        publisher.publish({"ttt_corners": corners.tolist()})

        commands = planner.plan_move(board, move, corners)
        print(f"Board {board}: playing {move}", flush=True)
        print("Moving right arm from side T-pose to board hover", flush=True)

        active_pose = None
        for index, command in enumerate(commands):
            if command[0] == "move":
                active_pose = index
                duration = 4.0
            else:
                duration = 1.5
                controller.gripper(command[1])
            deadline = None
            while deadline is None or time.monotonic() < deadline:
                target = commands[active_pose]
                ik_joints = controller.ik(target[1], target[2])
                smooth_joints += POSE_BLEND * (ik_joints - smooth_joints)
                if deadline is None:
                    state = state_subscriber.get_msg()
                    measured_waist = (
                        float(np.asarray(state["q"])[waist_index])
                        if state is not None and state.get("q") is not None
                        else float("nan")
                    )
                    print(
                        f"Waist at planner step {index}: "
                        f"command={np.degrees(smooth_joints[controller.waist_yaw]):.2f} deg, "
                        f"measured={np.degrees(measured_waist):.2f} deg",
                        flush=True,
                    )
                    deadline = time.monotonic() + duration
                send(smooth_joints, 1.0, corners)
                time.sleep(0.2)
        print("Returning from board-ready pose to side T-pose", flush=True)
        side_t = controller.side_t_pose(achieved_waist_yaw)
        finish = time.monotonic() + POSE_DURATION
        while time.monotonic() < finish:
            smooth_joints += POSE_BLEND * (side_t - smooth_joints)
            send(smooth_joints, 1.0, corners)
            time.sleep(0.2)
        send(side_t, 2.0, corners)
        time.sleep(2.0)
        print("Sending GR00T yaw key 7 four times", flush=True)
        press_policy_key("7", YAW_KEY_COUNT)
        send(
            controller.side_t_pose(0.0),
            POSE_DURATION,
            corners,
            preserve_waist=False,
        )
        time.sleep(POSE_DURATION)
        print("Tic-tac-toe move complete", flush=True)

    def start_move():
        nonlocal move_thread
        if move_thread is not None and move_thread.is_alive():
            print("A tic-tac-toe move is already running", flush=True)
            return
        move_thread = threading.Thread(target=run_move, daemon=True)
        move_thread.start()

    def start_in_side_t_pose():
        send(controller.side_t_pose(), POSE_DURATION)

    run_g1_control_loop(
        config,
        program_action=start_move,
        startup_action=start_in_side_t_pose,
    )


if __name__ == "__main__":
    main(tyro.cli(ControlLoopConfig))
