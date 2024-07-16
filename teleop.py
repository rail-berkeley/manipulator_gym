#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
from manipulator_gym.interfaces.interface_service import ActionClientInterface


def print_yellow(x): return print("\033[93m {}\033[00m" .format(x))


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
    print_yellow("    q: quit")


###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Teleoperation to a manipulator server')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--eef_displacement', type=float, default=0.01)
    parser.add_argument('--use_spacemouse', action='store_true')
    parser.add_argument('--no_rotation', action='store_true')
    args = parser.parse_args()

    interface = ActionClientInterface(host=args.ip, port=args.port)

    _ed = args.eef_displacement

    if args.use_spacemouse:
        print("Using SpaceMouse for teleoperation.")
        from manipulator_gym.utils.spacemouse import SpaceMouseExpert
        spacemouse = SpaceMouseExpert()

        def apply_spacemouse_action(gripper_open, with_rotation=True):
            sm_action, buttons = spacemouse.get_action()
            action = np.zeros(7)
            action[-1] = 1 if gripper_open else 0

            dim = 6 if with_rotation else 3
            for i in range(dim):
                if sm_action[i] > 0.5:
                    action[i] = _ed
                elif sm_action[i] < -0.5:
                    action[i] = -_ed
            interface.step_action(action)
    else:
        keyboard_action_map = {
            ord('w'): np.array([_ed, 0, 0, 0, 0, 0,  0]),
            ord('s'): np.array([-_ed, 0, 0, 0, 0, 0, 0]),
            ord('a'): np.array([0, _ed, 0, 0, 0, 0,  0]),
            ord('d'): np.array([0, -_ed, 0, 0, 0, 0, 0]),
            ord('z'): np.array([0, 0, _ed, 0, 0, 0,  0]),
            ord('c'): np.array([0, 0, -_ed, 0, 0, 0, 0]),
            ord('i'): np.array([0, 0, 0, _ed, 0, 0,  0]),
            ord('k'): np.array([0, 0, 0, -_ed, 0, 0, 0]),
            ord('j'): np.array([0, 0, 0, 0, _ed, 0,  0]),
            ord('l'): np.array([0, 0, 0, 0, -_ed, 0, 0]),
            ord('n'): np.array([0, 0, 0, 0, 0, _ed,  0]),
            ord('m'): np.array([0, 0, 0, 0, 0, -_ed, 0]),
        }

    print_help(not args.use_spacemouse)
    is_open = 1
    running = True
    while running:
        # Check for key press
        key = cv2.waitKey(100) & 0xFF

        # escape key to quit
        if key == ord('q'):
            print("Quitting teleoperation.")
            running = False
            continue

        # space bar to change gripper state
        elif key == ord(' '):
            is_open = 1 - is_open
            print("Gripper is now: ", is_open)
            interface.step_action(np.array([0, 0, 0, 0, 0, 0, is_open]))
        elif key == ord('r'):
            print("Resetting robot...")
            interface.reset()
            print_help()
        elif key == ord('g'):
            print("Going to sleep... make sure server has this method")
            kwargs = {"go_sleep": True}
            interface.reset(**kwargs)
            print_help()

        if args.use_spacemouse:
            # command robot with spacemouse
            apply_spacemouse_action(is_open, not args.no_rotation)
        elif key in keyboard_action_map:
            # command robot with keyboard
            action = keyboard_action_map[key]
            action[-1] = is_open
            interface.step_action(action)

        show_video(interface)

    cv2.destroyAllWindows()
    print("Teleoperation ended.")
