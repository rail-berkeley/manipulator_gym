import numpy as np
import math
import cv2

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

from manipulator_gym.interfaces.base_interface import ManipulatorInterface
from manipulator_gym.utils.utils import (
    rotationMatrixToEulerAngles,
    eulerAnglesToRotationMatrix,
)

##############################################################################


class WidowXRos2Interface(ManipulatorInterface):
    """
    Interbotix ROS2 api
    https://github.com/Interbotix/interbotix_ros_toolboxes/blob/humble/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot/arm.py
    """

    def __init__(self, cam_ids=[0], blocking_control=True):
        self._bot = InterbotixManipulatorXS("wx250s", "arm", "gripper")
        self._arm = self._bot.arm
        self._gripper = self._bot.gripper

        print("Using camera ids: ", cam_ids)
        self._caps = []
        for id in cam_ids:
            self._caps.append(
                cv2.VideoCapture(id)
            )  # Initialize the VideoCapture object
        assert len(self._caps) <= 2, "Only 2 cameras are supported"
        self.blocking_control = blocking_control
        # start persistent image fetching thread to avoid latency issues
        img_primary_thread = self.start_img_fetch_thread()

    def fetch_primary_img(self) -> None:
        ret, frame = self._caps[0].read()
        # If frame is read correctly ret is True
        if not ret:
            raise Exception("Can't receive frame (stream end?). Exiting ...")
        self._primary_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def fetch_wrist_img(self) -> None:
        if len(self._caps) < 2:
            self._wrist_frame = None
        else:
            ret, frame = self._caps[1].read()
            # If frame is read correctly ret is True
            if not ret:
                raise Exception("Can't receive frame (stream end?). Exiting ...")
            self._wrist_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @property
    def eef_pose(self) -> np.ndarray:
        return self._get_ee_pose()

    @property
    def gripper_state(self) -> float:
        # Exact ref implementation:
        # https://github.com/Interbotix/interbotix_ros_toolboxes/blob/0c739cdab1dbab03d79e752b43fa3db14d5bb15e/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/gripper.py#L62
        gripper_pos = self._gripper.core.joint_states.position[-3]
        return gripper_pos

    def step_action(self, action: np.ndarray) -> bool:
        """
        Override function from base class
        """
        # move the end effector, Not that dz is up and down, dy is left and right
        print("running action: ", action)
        action[:6] = np.clip(action[:6], -0.01, 0.01)
        self._move_eef_relative(
            dx=action[0],
            dy=action[1],
            dz=action[2],
            drx=action[3],
            dry=action[4],
            drz=action[5],
        )
        if action[6] > 0.5:
            print("open gripper", self.gripper_state)
            self.move_gripper(1.0)
        else:
            print("close gripper", self.gripper_state)
            self.move_gripper(0.0)
        return True

    def reset(self, reset_pose=True) -> bool:
        """Override function from base class"""
        print("Reset robot interface, reset to home pose?: ", reset_pose)
        if reset_pose:
            self.move_eef(np.array([0.258325, 0, 0.19065, 0, math.pi / 2, 0]))
            self._gripper.release()
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
        # time.sleep(0.05)

    def move_eef(self, pose: np.ndarray) -> bool:
        """
        takes in a 6D pose moves the arm to the target pose
        """
        assert len(pose) == 6, "pose is not 6D"
        H_mat = np.eye(4)
        H_mat[:3, 3] = pose[:3]
        H_mat[:3, :3] = eulerAnglesToRotationMatrix(pose[3], pose[4], pose[5])
        # NOTE: this is important before commanding the robot since
        # the interbotix code is bad due to persistent state, this avoids
        # the robot from moving too fast with insane accel and vel
        self._arm.capture_joint_positions()
        self._arm.set_trajectory_time(moving_time=4.0, accel_time=2.0)
        self._arm.set_ee_pose_matrix(H_mat, moving_time=4.0)
        self._move_eef_relative(0, 0, 0, 0, 0, 0)  # TODO FIx bug in compliance control
        return True

    def move_gripper(self, grip_state: float):
        if grip_state > 0.5 and self.gripper_state < 0.5:
            print("release")
            self._gripper.release(delay=1.5)
        elif grip_state < 0.5 and self.gripper_state > 0.5:
            print("grasp")
            self._gripper.grasp(delay=1.5)
        return True

    def _get_ee_pose(self):
        """This returns a np array of the x, y, z of the end effector"""
        T_sb = self._arm.get_ee_pose()
        xyz = T_sb[:3, 3]
        rpy = rotationMatrixToEulerAngles(T_sb[:3, :3])
        return np.concatenate([xyz, rpy])

    def __del__(self):
        del self._arm
        self._bot.shutdown()
