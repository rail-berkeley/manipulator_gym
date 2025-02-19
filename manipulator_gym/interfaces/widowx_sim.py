import numpy as np
import time
from typing import Optional
from manipulator_gym.utils.kinematics import KinematicsSolver

import os
import pybullet as p
import pybullet_data


class WidowXSimInterface:
    """
    Defines the base abstract class of WidowX Sim interface. This class
    is used for the gym env to interact with the Widowx sim env.
    """

    def __init__(
        self,
        default_pose=np.array([0.2, 0.0, 0.15, 0.0, 1.57, 0.0, 1.0]),
        image_size=(480, 640),
        headless=False,
    ):
        """
        Define the environment
        : args default_pose: 7-dimensional vector of
                            [x, y, z, roll, pitch, yaw, gripper_state]
        : args im_size: image size
        : args headless: whether to run the simulation in headless mode
        """
        assert len(default_pose) == 7
        self.image_size = image_size
        self.default_pose = default_pose
        # Initialize PyBullet, and hide side panel but still show obs
        self.client = p.connect(p.DIRECT if headless else p.GUI)

        # Disable shadow as it hurts performance of the policy since policy is not trained with this env
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.table = p.loadURDF("table/table.urdf", 0.5, 0.0, -0.63, 0.0, 0.0, 0.0, 1.0)

        # TODO: use the correct path to the asset
        asset_path = os.path.dirname(os.path.realpath(__file__)) + "/../utils/assets"

        # NOTE: Original URDF is from
        # https://github.com/avisingh599/roboverse/tree/master/roboverse/assets/interbotix_descriptions
        urdf_path = asset_path + "/widowx/urdf/wx250.urdf"
        eef_link = "/ee_gripper_link"
        self.arm = p.loadURDF(urdf_path, useFixedBase=True)
        link_name_to_index = self._find_bullet_link_names(self.arm)
        self.eef_link_id = link_name_to_index[eef_link]  # link id == 11

        for id in [9, 10, 11]:
            p.changeDynamics(self.arm, id, lateralFriction=100)

        # NOTE: users can add more objects to the scene by impl p.loadURDF()
        # https://github.com/ChenEating716/pybullet-URDF-models/tree/main
        # https://github.com/bulletphysics/bullet3/tree/master/data # with collision
        asset_specs = [
            (f"{asset_path}/red_marker/model.urdf", [0.32, -0.05, 0.01], 1.0),
            (f"{asset_path}/banana/model.urdf", [0.28, -0.13, 0.01], 0.8),
            (f"{asset_path}/blue_plate/model.urdf", [0.28, 0.12, 0.02], 1.0),
        ]
        for asset, pos, scale in asset_specs:
            p.loadURDF(asset, pos, globalScaling=scale)

        # Define camera parameters
        # https://docs.google.com/document/d/1si-6cTElTWTgflwcZRPfgHU7-UwfCUkEztkH3ge5CGc/edit
        # TODO: calibrate the intrinsic and extrinsic parameters
        camera_eye_position = [0.04, -0.12, 0.34]  # x=4cm, y=12cm, z=34cm
        camera_target_position = [0.38, 0, 0]  # Pointing at the origin
        camera_up_vector = [0, 0, 1]  # Z-axis up
        self.cam_view_matrix = p.computeViewMatrix(
            camera_eye_position, camera_target_position, camera_up_vector
        )
        self.cam_proj_matrix = p.computeProjectionMatrixFOV(
            fov=51.83,  # vertical fov, logitech C920 diagonal fov is 78.0
            aspect=image_size[1] / image_size[0],  # width/height
            nearVal=0.01,
            farVal=99.0,
        )
        self.ksolver = KinematicsSolver(urdf_path, eef_link=eef_link)
        p.setGravity(0, 0, -10)
        self.move_eef(self.default_pose[:6], reset=True)
        self.move_gripper(self.default_pose[-1], reset=True)

    @property
    def eef_pose(self) -> np.ndarray:
        """return the [x, y, z, rx, ry, rz] of the end effector"""
        # Get proprioceptive information, end-effector [x, y, z, roll, pitch, yaw]
        current_state = p.getLinkState(self.arm, self.eef_link_id)
        pos = current_state[0]
        orn = current_state[1]
        euler = p.getEulerFromQuaternion(orn)
        eef_pose = np.array(pos + euler)
        return eef_pose

    @property
    def gripper_state(self) -> float:
        """
        Return the gripper state
            1: open, 0: close
        """
        grip_joint_indices = [9, 10]
        grip_state = []
        for joint_id in grip_joint_indices:
            grip_state.append(abs(p.getJointState(self.arm, joint_id)[0]))
        return 1.0 if sum(grip_state) > 0.05 else 0.0

    @property
    def primary_img(self) -> np.ndarray:
        """return the image from the camera"""
        return self._primary_frame

    @property
    def wrist_img(self) -> Optional[np.ndarray]:
        """
        return the image from the wrist camera
        """
        return self._wrist_frame
    
    def fetch_primary_img(self) -> None:
        """return the image from the camera"""
        # Obtain the camera image
        img_arr = p.getCameraImage(
            height=self.image_size[0],
            width=self.image_size[1],
            viewMatrix=self.cam_view_matrix,
            projectionMatrix=self.cam_proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,  # Use hardware acceleration
        )[2]
        # reshape from (height, width, 4) to (height, width, 3)
        img_arr = img_arr[:, :, :3]
        img_arr = np.array(img_arr, dtype=np.uint8)
        # img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        self._primary_frame = img_arr

    def fetch_wrist_img(self, return_blank: bool = True) -> None:
        """
        Return the image from the wrist camera. Default blank image
        """
        if return_blank:
            """default return blank img"""
            self._wrist_frame = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        else:
            # NOTE: experimental feature to use wrist camera
            img_arr = p.getCameraImage(
                height=self.image_size[0],
                width=self.image_size[1],
                viewMatrix=self._compute_wrist_cam_view_matrix(),
                projectionMatrix=self.cam_proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,  # Use hardware acceleration
            )[2]
            img_arr = img_arr[:, :, :3]
            img_arr = np.array(img_arr, dtype=np.uint8)
            self._wrist_frame = img_arr

    def step_action(self, action: np.ndarray) -> bool:
        """
        step an relative action
        input array of (dx, dy, dz, drx, dry, drz, gripper)
        return True if done
        """
        start_time = time.time()
        # action[:6] = np.clip(action[:6], -0.03, 0.02)
        print("step ", [round(a, 2) for a in action])
        p.stepSimulation()
        # Get image and proprioceptive data
        proprio = np.concatenate([self.eef_pose, [self.gripper_state]])
        abs_action = proprio + action
        if abs_action[2] < 0.01:  # explicitly set the min z
            abs_action[2] = 0.01
        self.move_eef(abs_action[:6])
        if int(self.gripper_state) != round(action[-1]):
            print("gripper action: ", "open" if action[-1] > 0.5 else "close")
            # sticky gripper behavior
            for _ in range(15):
                p.stepSimulation()
                self.move_gripper(action[-1])
        else:
            self.move_gripper(action[-1])
        print("Time taken for step: ", time.time() - start_time)
        return True

    def move_eef(self, pose: np.ndarray, reset=False):
        """
        Move the endeffector to the target position and orientation
        : args action: 6-dimensional vector of absolute
                        [x, y, z, roll, pitch, yaw]
        : args reset: whether to reset the joint states else use position control
        """
        current_joints = []
        for i in range(5):
            current_joints.append(p.getJointState(self.arm, i)[0])

        joints = self.ksolver.ik(
            target_position=pose[:3],
            target_orientation=pose[3:6],
            initial_state=current_joints,
        )
        assert len(joints) == 5

        if reset:
            # reset joint states
            for i in range(5):
                p.resetJointState(self.arm, i, joints[i])
        else:
            # print("  target joints: ", [round(j, 2) for j in joints])
            # apply joint angles to the arm
            for i in range(5):
                p.setJointMotorControl2(
                    bodyIndex=self.arm,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joints[i],
                    force=1000,
                    maxVelocity=50.0,
                )
        return True

    def move_gripper(self, state: float, reset=False):
        """
        Move the gripper to a specific state
        : args state: 1: open, 0: close
        """
        grip_joint_indices = [9, 10]  # the joint indices of the gripper
        grip_position = [0.0, 0.0] if state < 0.5 else [0.04, -0.04]
        if reset:
            # reset joint states
            for i in range(2):
                p.resetJointState(self.arm, grip_joint_indices[i], grip_position[i])
        else:
            for i in range(2):
                p.setJointMotorControl2(
                    bodyIndex=self.arm,
                    jointIndex=grip_joint_indices[i],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=grip_position[i],
                    force=1000,
                    maxVelocity=50.0,
                )
        return True

    def reset(self, reset_pose=True) -> bool:
        """Override function from base class"""
        print("Reset robot interface, reset to home pose?: ", reset_pose)
        if reset_pose:
            p.setGravity(0, 0, -10)
            self.move_eef(self.default_pose[:6], reset=True)
            self.move_gripper(self.default_pose[-1], reset=True)
        return True

    def _compute_wrist_cam_view_matrix(self):
        """Compute the view matrix for the wrist camera."""
        # Get the current position and orientation of the end effector
        wrist_cam_pose = self.eef_pose
        rotation_matrix = p.getMatrixFromQuaternion(
            p.getQuaternionFromEuler(wrist_cam_pose[3:])
        )
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
        camera_eye_position = wrist_cam_pose[:3]  # x, y, z
        # move camera_eye_position 0.1 along the z-axis after rotation
        camera_eye_position = camera_eye_position + np.dot(
            rotation_matrix, np.array([0, 0, 0.1])
        )

        # get the camera target position 0.1 meters in front of the camera
        camera_target_position = camera_eye_position + np.dot(
            rotation_matrix, np.array([1, 0, 0])
        )
        camera_up_vector = np.dot(rotation_matrix, np.array([0, 0, 1]))
        # Compute and return the view matrix
        return p.computeViewMatrix(
            camera_eye_position, camera_target_position, camera_up_vector
        )

    def _find_bullet_link_names(self, model_id):
        """utils to find the link names in pybullet"""
        _link_name_to_index = {
            p.getBodyInfo(model_id)[0].decode("UTF-8"): -1,
        }

        for _id in range(p.getNumJoints(model_id)):
            _name = p.getJointInfo(model_id, _id)[12].decode("UTF-8")
            _link_name_to_index[_name] = _id
        return _link_name_to_index

    def __del__(self):
        p.disconnect(self.client)
