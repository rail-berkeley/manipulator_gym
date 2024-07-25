#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
from manipulator_gym.interfaces.interface_service import ActionClientInterface


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def show_video(interface):
    """
    This shows the video from the camera for a given duration.
    """
    img = interface.primary_img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("primary img", img)

    wrist_img = interface.wrist_img
    if wrist_img is not None:
        wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("wrist img", wrist_img)

    cv2.waitKey(20)  # 20 ms


def print_help(with_keyboard=True):
    print_yellow("  Teleop Controls:")

    if with_keyboard:
        print_yellow("    w, s : move forward/backward")
        print_yellow("    a, d : move left/right")
        print_yellow("    z, c : move up/down")
        print_yellow("    i, k:  rotate yaw")
        print_yellow("    j, l:  rotate pitch")
        print_yellow("    n  m:  rotate roll")
    else:
        print_yellow("    SpaceMouse control [x, y, z, rx, ry, rz]")

    print_yellow("    space: toggle gripper")
    print_yellow("    r: reset robot")
    print_yellow("    g: go to sleep")
    print_yellow("    /: reboot mulfuction motor [experimental]")
    print_yellow("    q: quit")


###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Teleoperation to a manipulator server"
    )
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--eef_displacement", type=float, default=0.01)
    parser.add_argument("--use_spacemouse", action="store_true")
    parser.add_argument("--no_rotation", action="store_true")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--log_lang_text", type=str, default="null task")
    args = parser.parse_args()

    interface = ActionClientInterface(host=args.ip, port=args.port)

    _ed = args.eef_displacement

    if args.use_spacemouse:
        print("Using SpaceMouse for teleoperation.")
        from manipulator_gym.utils.spacemouse import SpaceMouseExpert

        spacemouse = SpaceMouseExpert()

        def _get_spacemouse_action(gripper_open, with_rotation=True):
            sm_action, buttons = spacemouse.get_action()
            action = np.zeros(7)
            action[-1] = 1 if gripper_open else 0

            dim = 6 if with_rotation else 3
            for i in range(dim):
                if sm_action[i] > 0.5:
                    action[i] = _ed
                elif sm_action[i] < -0.5:
                    action[i] = -_ed
            return action

    else:
        keyboard_action_map = {
            ord("w"): np.array([_ed, 0, 0, 0, 0, 0, 0]),
            ord("s"): np.array([-_ed, 0, 0, 0, 0, 0, 0]),
            ord("a"): np.array([0, _ed, 0, 0, 0, 0, 0]),
            ord("d"): np.array([0, -_ed, 0, 0, 0, 0, 0]),
            ord("z"): np.array([0, 0, _ed, 0, 0, 0, 0]),
            ord("c"): np.array([0, 0, -_ed, 0, 0, 0, 0]),
            ord("i"): np.array([0, 0, 0, _ed, 0, 0, 0]),
            ord("k"): np.array([0, 0, 0, -_ed, 0, 0, 0]),
            ord("j"): np.array([0, 0, 0, 0, _ed, 0, 0]),
            ord("l"): np.array([0, 0, 0, 0, -_ed, 0, 0]),
            ord("n"): np.array([0, 0, 0, 0, 0, _ed, 0]),
            ord("m"): np.array([0, 0, 0, 0, 0, -_ed, 0]),
        }


    def _get_full_obs():
        return {
            "image_primary": interface.primary_img,
            "image_wrist": interface.wrist_img,
            "state": np.concatenate([
                interface.eef_pose[:6],
                [0.0],  # padding
                [interface.gripper_state]], dtype=np.float32
            )
        }

    if args.log_dir:
        from oxe_envlogger.data_type import get_gym_space
        from oxe_envlogger.rlds_logger import RLDSLogger, RLDSStepType
        import tensorflow_datasets as tfds

        # Create RLDSLogger
        logger = RLDSLogger(
            observation_space=get_gym_space(_get_full_obs()),
            action_space=get_gym_space(np.zeros(7, dtype=np.float32)),
            dataset_name="test_rlds",
            directory=args.log_dir,
            max_episodes_per_file=1,
            step_metadata_info={"language_text": tfds.features.Text()},
        )
        _mdata = {"language_text": args.log_lang_text}

    def _execute_action(action, first_step=False):
        interface.step_action(action)
        if args.log_dir:
            obs = _get_full_obs()
            step_type = RLDSStepType.FIRST if first_step else RLDSStepType.TRANSITION
            logger(action, obs, 0.0, metadata=_mdata, step_type=step_type)

    def _execute_reset():
        null_action = np.zeros(7)
        if args.log_dir:
            obs = _get_full_obs()
            logger(null_action, obs, 1.0, metadata=_mdata, step_type=RLDSStepType.TERMINATION)

        interface.reset()

        if args.log_dir:
            obs = _get_full_obs()
            logger(null_action, obs, 0.0, metadata=_mdata, step_type=RLDSStepType.RESTART)            

    print_help(not args.use_spacemouse)
    is_open = 1
    running = True

    _execute_action(np.array([0, 0, 0, 0, 0, 0, is_open]), first_step=True)

    while running:
        # Check for key press
        key = cv2.waitKey(100) & 0xFF

        # escape key to quit
        if key == ord("q"):
            print("Quitting teleoperation.")
            running = False
            continue

        # space bar to change gripper state
        elif key == ord(" "):
            is_open = 1 - is_open
            print("Gripper is now: ", is_open)
            _execute_action(np.array([0, 0, 0, 0, 0, 0, is_open]))
        elif key == ord("r"):
            print("Resetting robot...")
            _execute_reset()
            print_help()
        elif key == ord("g"):
            print("Going to sleep... make sure server has this method")
            kwargs = {"go_sleep": True}
            interface.reset(**kwargs)
            print_help()
        elif key == ord("/"):
            print("[experimental feature] reboot mulfuction motor for widowx")
            widowx_joints = [
                "waist",
                "shoulder",
                "elbow",
                "forearm_roll",
                "wrist_angle",
                "wrist_rotate",
                "gripper",
            ]
            res = interface.custom_fn("motor_status")
            print("Motor status: ", res)

            for i, status in enumerate(res):
                if status != 0:
                    joint_name = widowx_joints[i]
                    print("Rebooting motor: ", joint_name)
                    interface.custom_fn("reboot_motor", joint_name=joint_name)

            print_help()

        if args.use_spacemouse:
            # command robot with spacemouse
            action = _get_spacemouse_action(is_open, not args.no_rotation)

            # if action is more than 0.001 or less than -0.001 then move
            if np.any(action[:6] > 0.001) or np.any(action[:6] < -0.001):
                _execute_action(action)

        elif key in keyboard_action_map:
            # command robot with keyboard
            action = keyboard_action_map[key]
            action[-1] = is_open
            _execute_action(action)

        show_video(interface)

    if args.log_dir:
        logger.close()
        print("Done logging.")

    cv2.destroyAllWindows()
    print("Teleoperation ended.")
