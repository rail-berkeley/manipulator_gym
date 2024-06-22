import rospy
from sensor_msgs.msg import Image

import numpy as np
import math

from manipulator_gym.interfaces.base_interface import ManipulatorInterface
from manipulator_gym.utils.utils import convert_img, \
    rotationMatrixToEulerAngles, eulerAnglesToRotationMatrix

from interbotix_xs_modules.arm import InterbotixManipulatorXS


np.set_printoptions(precision=3, suppress=True)

##############################################################################

class ViperXInterface(ManipulatorInterface):
    """
    https://github.com/Interbotix/interbotix_ros_toolboxes/blob/main/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/arm.py   
    
    For camera images, we expect the images are published to the
    following topics via ROS:
      - "/gripper_cam/image_raw"
      - "/third_perp_cam/image_raw"

    else: overload the impl to get the images directly via cv2.VideoCapture
    """

    def __init__(self, init_node=True, blocking_control=True):
        self._bot = InterbotixManipulatorXS(
            "vx300s", "arm", "gripper", init_node=init_node)
        self._arm = self._bot.arm
        self._gripper = self._bot.gripper
        self._cam_sub1 = rospy.Subscriber(
            "/gripper_cam/image_raw", Image, self._update_wrist_cam)
        self._cam_sub2 = rospy.Subscriber(
            "/third_perp_cam/image_raw", Image, self._update_primary_cam)
        self._side_img = np.array((480, 640, 3), dtype=np.uint8)
        self._wrist_img = np.array((480, 640, 3), dtype=np.uint8)
        self.blocking_control = blocking_control

    @property
    def primary_img(self) -> np.ndarray:
        return self._side_img

    @property
    def wrist_img(self):
        return self._wrist_img

    @property
    def eef_pose(self) -> np.ndarray:
        return self._get_ee_pose()

    @property
    def gripper_state(self) -> float:
        # Exact ref implementation:
        # https://github.com/Interbotix/interbotix_ros_toolboxes/blob/0c739cdab1dbab03d79e752b43fa3db14d5bb15e/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/gripper.py#L62
        gripper_pos = \
            self._gripper.core.joint_states.position[self._gripper.left_finger_index]
        # NOTE: explicitly scale the value of the gripper pos
        gripper_pos = (gripper_pos - 0.01) / (0.08 - 0.01)
        return gripper_pos

    # override
    def step_action(self, action: np.ndarray) -> bool:
        """
        Override function from base class
        """
        # move the end effector, Not that dz is up and down, dy is left and right
        self._move_eef_relative(dx=action[0],
                                dy=action[1],
                                dz=action[2],
                                drx=action[3],
                                dry=action[4],
                                drz=action[5])
        if action[6] > 0.5:
            self._gripper.open(delay=0.1)
        else:
            print("close gripper")
            self._gripper.close(delay=0.1)
        print("action", action)
        return True

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
        return True

    def move_gripper(self, grip_state: float):
        if grip_state > 0.5:
            self._gripper.open(delay=0.1)
        else:
            self._gripper.close(delay=0.1)
        return True

    def reset(self,
              reset_pose=True,
              target_state=np.array([0.26, 0.0, 0.26, 0.0, math.pi/2, 0.0, 0.0]),
              go_sleep=False,
        ) -> bool:
        """Override function from base class"""
        print("Reset robot interface, reset to home pose?: ", reset_pose)
        if reset_pose:
            print("Moving to target state: ", target_state)
            if target_state[6] > 0.5:
                self._gripper.open(delay=0.1)
            else:
                self._gripper.close(delay=0.1)
            self.move_eef(target_state[:6])
            if go_sleep:
                self._arm.go_to_sleep_pose()
        return True

    def _update_primary_cam(self, img_msg):
        # convert img_msg to numpy array
        self._side_img = convert_img(img_msg)

    def _update_wrist_cam(self, img_msg):
        # convert img_msg to numpy array
        self._wrist_img = convert_img(img_msg)

    def _move_eef_relative(self, dx=0, dy=0, dz=0, drx=0, dry=0, drz=0):
        curr_rpy = rotationMatrixToEulerAngles(self._arm.T_sb[:3, :3])
        self._arm.set_ee_pose_components(
            x=self._arm.T_sb[0, 3] + dx,
            y=self._arm.T_sb[1, 3] + dy,
            z=self._arm.T_sb[2, 3] + dz,
            roll=curr_rpy[0] + drx,
            pitch=curr_rpy[1] + dry,
            yaw=curr_rpy[2] + drz,
            moving_time=0.2,  # TODO: make this configurable
            blocking=self.blocking_control,
            custom_guess=self._arm.get_joint_commands(),
        )

    def _get_ee_pose(self):
        """This returns a np array of the x, y, z of the end effector"""
        T_sb = self._arm.get_ee_pose()
        xyz = T_sb[:3, 3]
        rpy = rotationMatrixToEulerAngles(T_sb[:3, :3])
        return np.concatenate([xyz, rpy])
