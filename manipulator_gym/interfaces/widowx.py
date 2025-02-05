import rospy
from sensor_msgs.msg import Image

import numpy as np
import math
import cv2

from manipulator_gym.utils.utils import rotationMatrixToEulerAngles
from manipulator_gym.interfaces.viperx import ViperXInterface
from interbotix_xs_modules.arm import InterbotixManipulatorXS


##############################################################################

class WidowXInterface(ViperXInterface):
    """
    https://github.com/Interbotix/interbotix_ros_toolboxes/blob/main/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/arm.py
    """

    def __init__(self, init_node=True, blocking_control=True, cam_ids=None):
        """
        Args:
            cam_ids (list): list of camera ids to use for the interface,
                if None, use ROS subscribers to get images
        """
        self._bot = InterbotixManipulatorXS(
            "wx250s", "arm", "gripper", init_node=init_node
        )
        self._arm = self._bot.arm
        self._gripper = self._bot.gripper

        # user have the option to use rossub or native opencv to read camera
        if cam_ids is None:
            self._cam_sub1 = rospy.Subscriber(
                "/gripper_cam/image_raw", Image, self._update_wrist_cam
            )
            self._cam_sub2 = rospy.Subscriber(
                "/third_perp_cam/image_raw", Image, self._update_primary_cam
            )
            self._caps = None
        else:
            print("Using camera ids: ", cam_ids)
            assert len(cam_ids) <= 2, "Only 2 cameras are supported"
            self._caps = []
            for id in cam_ids:
                self._caps.append(cv2.VideoCapture(id))

        self._side_img = np.array((480, 640, 3), dtype=np.uint8)
        self._wrist_img = np.array((480, 640, 3), dtype=np.uint8)
        self.blocking_control = blocking_control
        # start persistent image fetching thread to avoid latency issues
        img_primary_thread = self.start_img_fetch_thread()

    @property
    def primary_img(self) -> np.ndarray:
        if self._caps is None:
            return self._side_img
        else:
            ret, frame = self._caps[0].read()
            # If frame is read correctly ret is True
            if not ret:
                raise Exception("Can't receive frame (stream end?). Exiting ...")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @property
    def wrist_img(self):
        if self._caps is None:
            return self._wrist_img
        elif len(self._caps) < 2:  # no idx 1
            return None
        else:
            ret, frame = self._caps[1].read()
            # If frame is read correctly ret is True
            if not ret:
                raise Exception("Can't receive frame (stream end?). Exiting ...")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def reset(
        self,
        reset_pose=True,
        target_state=np.array(np.array([0.258325, 0, 0.19065, 0, math.pi / 2, 0, 1.0])),
        go_sleep=False,
        moving_time=None,
    ) -> bool:
        """
        Override function from base class
        Default reset position is home pose, with open gripper
        """
        print("Reset robot interface, reset to home pose?: ", reset_pose)
        if reset_pose:
            print("Moving to target state: ", target_state)
            if target_state[6] > 0.5:
                self._gripper.open(delay=0.1)
            else:
                self._gripper.close(delay=0.1)
            self.move_eef(target_state[:6])
            if go_sleep:
                self._arm.go_to_sleep_pose(moving_time=moving_time)
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

    def move_gripper(self, grip_state: float):
        """WidowX gripper state is between 0 and 0.39 (fully open)"""
        if grip_state > 0.25:
            self._gripper.open(delay=0.1)
        else:
            self._gripper.close(delay=0.1)
        return True

    def motor_status(self) -> np.ndarray:
        """
        Check if there are any hardware errors

        non-zero value means error

        API is from:
        https://github.com/Interbotix/interbotix_ros_toolboxes/blob/noetic/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/core.py
        """
        values = self._bot.dxl.robot_get_motor_registers(
            "group", "all", "Hardware_Error_Status"
        ).values
        int_value = np.array(values, dtype=np.uint8)
        try:
            assert len(values) == 7, "Expecting 7 joints"
        except AssertionError:
            # ignore this error because sometimes joints fail and return less than 7 values
            print("Error: motor status is", values)
        return int_value

    def reboot_motor(self, joint_name: str):
        """
        Reboot a single motor, provide the joint name

        Supported joint names:
            - waist, shoulder, elbow, forearm_roll,
            - wrist_angle, wrist_rotate, gripper, left_finger, right_finger
        """
        res = self._bot.dxl.robot_reboot_motors(
            "single", joint_name, enable=True, smart_reboot=True
        )
        return res

    def safe_reboot_all_motors(self, go_sleep=True, moving_time=5):
        """
        Optionally put the robot to sleep, then reboot all motors.
        """
        print("Going to sleep and rebooting all motors...")
        if go_sleep:
            self._gripper.open(delay=0.1)
            self._arm.go_to_sleep_pose(moving_time=moving_time)
        res = self._bot.dxl.robot_reboot_motors("group", "all", enable=True)
        return res

    def get_torque_status(self):
        """
        Get the torque status of all Dynamixel motors

        API is from:
        https://github.com/Interbotix/interbotix_ros_toolboxes/blob/53443a3d915db12d425364b319f90df3db3dddbe/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/core.py#L99
        Potential register values from:
        https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/

        Get a list of 7 values, each is either 0 (disabled) or 1 (enabled)
        """
        values = self._bot.dxl.robot_get_motor_registers(
            "group", "all", "Torque_Enable"
        ).values
        try:
            assert len(values) == 7, "Expecting 7 joints"
        except AssertionError:
            # ignore this error because sometimes motors fail and return less than 7 values
            print("Error: torque status is", values)
        values = np.array(values, dtype=np.uint8)
        return values

    def enable_torque(self, enable: bool = True):
        """
        Enable or disable the torque for all motors. No returns.

        API is from:
        https://github.com/Interbotix/interbotix_ros_toolboxes/blob/53443a3d915db12d425364b319f90df3db3dddbe/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/core.py#L115
        """
        print("Enabling torque: ", enable)
        self._bot.dxl.robot_torque_enable("group", "all", enable)

    def get_joint_status(self):
        """
        Get the joint states (position, velocity, effort) of all Dynamixel motors

        API is from:
        https://github.com/Interbotix/interbotix_ros_toolboxes/blob/53443a3d915db12d425364b319f90df3db3dddbe/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/core.py#L167-L173

        Returns <class 'sensor_msgs.msg._JointState.JointState'>
        e.g.
            header:
            seq: 265873
            stamp:
                secs: 1733509754
                nsecs: 197535524
            frame_id: ''
            name:
            - waist
            - shoulder
            - elbow
            - forearm_roll
            - wrist_angle
            - wrist_rotate
            - gripper
            - left_finger
            - right_finger
            position: [-0.4218447208404541, -0.40957286953926086, 0.48780590295791626, -0.003067961661145091, 1.529378890991211, -0.42337870597839355, 1.3652429580688477, 0.037534553557634354, -0.037534553557634354]
            velocity: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            effort: [-2.690000057220459, -32.279998779296875, -433.0899963378906, 45.72999954223633, 0.0, 0.0, 0.0, 0.0, 0.0]
        """
        values = self._bot.dxl.robot_get_joint_states()
        return values

    def joint_efforts(self) -> dict:
        """
        Get the joint efforts of all Dynamixel motors

        Returns:
            dict: joint_name -> effort
        """
        values = self.get_joint_status()
        return {name: effort for name, effort in zip(values.name, values.effort)}
