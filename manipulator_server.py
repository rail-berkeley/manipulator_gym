import logging
import argparse

from manipulator_gym.interfaces.interface_service import ManipulatorInterfaceServer
from manipulator_gym.interfaces.base_interface import ManipulatorInterface

if __name__ == "__main__":
    """
    Runs the ManipulatorServer on a robot machine with rospy installed.
    When running on the actual ros robot, this requires the bringup of 
        - interbotix_xsarm_control)/launch/xsarm_control.launch
        - usb camara published to /gripper_cam and /third_perp_cam
    """
    parser = argparse.ArgumentParser(description='Manipulator Server')
    parser.add_argument('--viperx', action='store_true',
                        help='run the environment with the ViperX')
    parser.add_argument('--widowx', action='store_true',
                        help='run the environment with the WidowX')
    parser.add_argument('--widowx_sim', action='store_true',
                        help='run the environment with the WidowX')
    parser.add_argument('--widowx_ros2', action='store_true',
                        help='run the environment with the WidowX ros2')
    parser.add_argument('--cam_id', type=int, default=0,
                        help='camera id it is used in the interface')
    parser.add_argument('--non_blocking', action='store_true',
                        help='run the environment with non_blocking control')
    args = parser.parse_args()

    # set logging to warning
    logging.basicConfig(level=logging.WARNING)
    _blocking = not args.non_blocking # default is blocking

    if args.viperx:
        from manipulator_gym.interfaces.viperx import ViperXInterface
        print("Using ViperX interface")
        interface = ViperXInterface(init_node=True, blocking_control=_blocking)
    elif args.widowx:
        from manipulator_gym.interfaces.widowx import WidowXInterface
        print("Using WidowX interface")
        blocking = not args.non_blocking
        interface = WidowXInterface(init_node=True, blocking_control=_blocking)
    elif args.widowx_sim:
        from manipulator_gym.interfaces.widowx_sim import WidowXSimInterface
        print("Using WidowXSim interface")
        interface = WidowXSimInterface()
    elif args.widowx_ros2:
        from manipulator_gym.interfaces.widowx_ros2 import WidowXRos2Interface
        print("Using WidowXRos2 interface")
        interface = WidowXRos2Interface(cam_id=args.cam_id, blocking_control=_blocking)
    else:
        print("Using base test interface")
        interface = ManipulatorInterface()

    # start the agentlace server
    server = ManipulatorInterfaceServer(
        manipulator_interface=interface,
    )

    server.start()
    server.stop()
    print("Server stopped")
