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
    parser = argparse.ArgumentParser(description="Manipulator Server")
    parser.add_argument(
        "--viperx", action="store_true", help="run the environment with the ViperX"
    )
    parser.add_argument(
        "--widowx", action="store_true", help="run the environment with the WidowX"
    )
    parser.add_argument(
        "--widowx_sim", action="store_true", help="run the environment with the WidowX"
    )
    parser.add_argument(
        "--widowx_ros2",
        action="store_true",
        help="run the environment with the WidowX ros2",
    )
    parser.add_argument(
        "--cam_ids",
        type=int,
        nargs="+",
        default=[],
        help="camera ids to use for the interface",
    )
    parser.add_argument(
        "--non_blocking",
        action="store_true",
        help="run the environment with non_blocking control",
    )
    parser.add_argument(
        "--resize_img",
        type=int,
        nargs=2,
        default=None,
        help="resize the image before sending to the client",
    )
    args = parser.parse_args()

    # set logging to warning
    logging.basicConfig(level=logging.WARNING)
    _blocking = not args.non_blocking  # default is blocking

    if args.viperx:
        from manipulator_gym.interfaces.viperx import ViperXInterface

        print("Using ViperX interface")
        interface = ViperXInterface(init_node=True, blocking_control=_blocking)
    elif args.widowx:
        from manipulator_gym.interfaces.widowx import WidowXInterface

        print("Using WidowX interface")
        blocking = not args.non_blocking
        cam_ids = args.cam_ids if len(args.cam_ids) > 0 else None
        interface = WidowXInterface(
            init_node=True, blocking_control=_blocking, cam_ids=cam_ids
        )
    elif args.widowx_sim:
        from manipulator_gym.interfaces.widowx_sim import WidowXSimInterface

        print("Using WidowXSim interface")
        interface = WidowXSimInterface()
    elif args.widowx_ros2:
        from manipulator_gym.interfaces.widowx_ros2 import WidowXRos2Interface

        print("Using WidowXRos2 interface")
        assert 0 < len(args.cam_ids) <= 2, "should provide 1 or 2 camera ids"
        interface = WidowXRos2Interface(
            cam_ids=args.cam_ids, blocking_control=_blocking
        )
    else:
        print("Using base test interface")
        interface = ManipulatorInterface()

    # start the agentlace server
    server = ManipulatorInterfaceServer(
        manipulator_interface=interface,
        resize_img=args.resize_img,
    )

    server.start()
    server.stop()
    print("Server stopped")
