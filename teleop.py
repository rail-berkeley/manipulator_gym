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


def print_help():
    print_yellow("  Teleop Controls:")
    print_yellow("    w, s : move forward/backward")
    print_yellow("    a, d : move left/right")
    print_yellow("    z, c : move up/down")
    print_yellow("    i, k:  rotate yaw")
    print_yellow("    j, l:  rotate pitch")
    print_yellow("    n  m:  rotate roll")
    print_yellow("    space: toggle gripper")
    print_yellow("    r: reset robot")
    print_yellow("    g: go to sleep")
    print_yellow("    q: quit")


def main():
    parser = argparse.ArgumentParser(
        description='Teleoperation to a manipulator server')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    args = parser.parse_args()

    interface = ActionClientInterface(host=args.ip, port=args.port)

    print_help()
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

        # Handle key press for robot control
        # translation
        if key == ord('w'):
            interface.step_action(np.array([0.01, 0, 0, 0, 0, 0, is_open]))
        elif key == ord('s'):
            interface.step_action(np.array([-0.01, 0, 0, 0, 0, 0, is_open]))
        elif key == ord('a'):
            interface.step_action(np.array([0, 0.01, 0, 0, 0, 0, is_open]))
        elif key == ord('d'):
            interface.step_action(np.array([0, -0.01, 0, 0, 0, 0, is_open]))
        elif key == ord('z'):
            interface.step_action(np.array([0, 0, 0.01, 0, 0, 0, is_open]))
        elif key == ord('c'):
            interface.step_action(np.array([0, 0, -0.01, 0, 0, 0, is_open]))

        # rotation
        elif key == ord('i'):
            interface.step_action(np.array([0, 0, 0, 0.01, 0, 0, is_open]))
        elif key == ord('k'):
            interface.step_action(np.array([0, 0, 0, -0.01, 0, 0, is_open]))
        elif key == ord('j'):
            interface.step_action(np.array([0, 0, 0, 0, 0.01, 0, is_open]))
        elif key == ord('l'):
            interface.step_action(np.array([0, 0, 0, 0, -0.01, 0, is_open]))
        elif key == ord('n'):
            interface.step_action(np.array([0, 0, 0, 0, 0, 0.01, is_open]))
        elif interface == ord('m'):
            interface.step_action(np.array([0, 0, 0, 0, 0, -0.01, is_open]))

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

        show_video(interface)

    cv2.destroyAllWindows()
    print("Teleoperation ended.")


if __name__ == "__main__":
    main()
