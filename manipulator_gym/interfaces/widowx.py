import rospy
from sensor_msgs.msg import Image

import numpy as np
import math

from manipulator_gym.utils.utils import rotationMatrixToEulerAngles
from manipulator_gym.interfaces.viperx import ViperXInterface

from interbotix_xs_modules.arm import InterbotixManipulatorXS


##############################################################################

class WidowXInterface(ViperXInterface):
    """
    https://github.com/Interbotix/interbotix_ros_toolboxes/blob/main/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/arm.py   
    """

    def __init__(self, init_node=True, blocking_control=True):
        self._bot = InterbotixManipulatorXS(
            "wx250s", "arm", "gripper", init_node=init_node)
        self._arm = self._bot.arm
        self._gripper = self._bot.gripper
        self._cam_sub1 = rospy.Subscriber(
            "/gripper_cam/image_raw", Image, self._update_wrist_cam)
        self._cam_sub2 = rospy.Subscriber(
            "/third_perp_cam/image_raw", Image, self._update_primary_cam)
        self._side_img = np.array((480, 640, 3), dtype=np.uint8)
        self._wrist_img = np.array((480, 640, 3), dtype=np.uint8)
        self.blocking_control = blocking_control

    def reset(self) -> bool:
        """Override function from base class and viperx class"""
        print("Resetting the robot")
        self.move_eef(np.array([0.258325, 0, 0.19065, 0, math.pi/2, 0]))
        self._gripper.open()
        return True

    def _move_eef_relative(self, dx=0, dy=0, dz=0, drx=0, dry=0, drz=0):
        curr_rpy = rotationMatrixToEulerAngles(self._arm.T_sb[:3, :3])
        self._arm.set_ee_pose_components(
            x=self._arm.T_sb[0, 3] + dx,
            y=self._arm.T_sb[1, 3] + dy,
            z=self._arm.T_sb[2, 3] + dz,
            roll=curr_rpy[0] + drx,
            pitch=curr_rpy[1] + dry,
            yaw=curr_rpy[2] + drz,
            moving_time=0.1,  # TODO: make this configurable
            blocking=self.blocking_control,
            custom_guess=self._arm.get_joint_commands(),
        )
