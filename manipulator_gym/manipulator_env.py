import gym
# import gymnasium as gym

import numpy as np
import argparse
import time
import cv2
from manipulator_gym.interfaces.base_interface import ManipulatorInterface
from manipulator_gym.interfaces.interface_service import ActionClientInterface
from enum import IntEnum
from typing import Optional, Dict, Callable


class StateEncoding(IntEnum):
    """Defines supported proprio state encoding schemes for different datasets.
    NOTE: this is defined in octo: oxe_dataset_configs.py
     ref: https://github.com/octo-models/octo/blob/8559a7077b266195c5611c7748a28aae2278def5/octo/data/oxe/oxe_dataset_configs.py#L26

    Current doesn't support JOINT_BIMANUAL and POS_QUAT (consider converting to POS_EULER)
    """
    NONE = -1  # no state provided
    POS_EULER = 1  # EEF XYZ + roll-pitch-yaw + 1 x pad + gripper open/close
    JOINT = 3  # 7 x joint angles (padding added if fewer) + gripper open/close
    # POS_QUAT = 2  # EEF XYZ + quaternion + gripper open/close
    # JOINT_BIMANUAL = 4  # 2 x [6 x joint angles + gripper open/close]


class ManipulatorEnv(gym.Env):
    def __init__(
        self,
        manipulator_interface: ManipulatorInterface,
        workspace_boundary=np.array([[-9., -9., -9.],  [9., 9., 9.]]),
        state_encoding=StateEncoding.POS_EULER,
        use_wrist_cam: bool = False,
        reward_fn: Optional[Callable[[Dict], float]] = None,
        done_fn: Optional[Callable[[Dict], tuple]] = None,
        eef_displacement : float = 0.02,
        out_of_boundary_penalty: float = 0.0,
    ):
        """
        Args:
        - manipulator_interface: interface to the manipulator
        - workspace_boundary: the boundary of the eef workspace in abs coordinates
        - state_encoding: the state encoding of the observation
        - use_wrist_cam: whether to use the wrist camera
        - reward_fn: reward function: Given an obs return a float of the reward
        - done_fn: function to check if the episode should terminate or truncate. return (term, trunc)
        - eef_displacement: the displacement of the eef in the action space
        - out_of_boundary_penalty: penalty for going out of the boundary (-ve reward)
           (this should be a negative value.)

        We would define the action space as a such [dx, dy, dz, drx, dry, drz, abs gripper]

        The observation space would be a dictionary containing
         - state: according to the state encoding
         - image_primary: rgb image of the camera
         - image_wrist: rgb image of the wrist camera (if available)
        """
        # simple assertion to check if things are correct
        assert state_encoding in StateEncoding, f"Invalid state encoding: {state_encoding}"
        if state_encoding == StateEncoding.JOINT:
            assert manipulator_interface.joint_states is not None, "Joint states not available"

        _sample_img_shape = manipulator_interface.primary_img.shape
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=np.zeros(_sample_img_shape, dtype=np.uint8),
                    high=255 * np.ones(_sample_img_shape, dtype=np.uint8),
                    dtype=np.uint8,
                ),
            }
        )

        if state_encoding is not StateEncoding.NONE:
            # NOTE: internally octo is using the key "state" for proprioception
            self.observation_space.spaces["state"] = gym.spaces.Box(
                low=np.ones((8,)) * -4, high=np.ones((8,))*4, dtype=np.float32
            )

        if use_wrist_cam and manipulator_interface.wrist_img is not None:
            _sample_img_shape = manipulator_interface.wrist_img.shape
            self.observation_space.spaces["image_wrist"] = gym.spaces.Box(
                low=np.zeros(_sample_img_shape, dtype=np.uint8),
                high=255 * np.ones(_sample_img_shape, dtype=np.uint8),
                dtype=np.uint8,
            )

        self._eef_displacement = eef_displacement
        self.action_space = gym.spaces.Box(
            low=np.ones(manipulator_interface.step_action_shape)*-eef_displacement,
            high=np.ones(manipulator_interface.step_action_shape)*eef_displacement,
            dtype=np.float32
        )

        print("action space: ", self.action_space)
        print("observation space: ", self.observation_space)
        print("-"*50)
        self._reward_fn = reward_fn
        self._done_fn = done_fn
        self._state_encoding = state_encoding
        self._use_wrist_cam = use_wrist_cam
        self.manipulator_interface = manipulator_interface
        
        # TODO: move this out as another gym.wrapper
        self.workspace_boundary = np.array(workspace_boundary)
        self.out_of_boundary_penalty = out_of_boundary_penalty

    def step(self, action: np.ndarray) -> tuple:
        obs = self._get_obs()

        # Default values
        terminal = False
        trunc = False
        reward = 0.0 if self._reward_fn is None else self._reward_fn(obs)

        # Handle Robot Actions
        if 'state' in obs:
            clip_low = self.workspace_boundary[0] - obs['state'][0:3]
            clip_high = self.workspace_boundary[1] - obs['state'][0:3]

            if np.any(clip_low > 0) or np.any(clip_high < 0):
                print("Warning: Action out of bounds. Clipping to workspace boundary.")
                reward -= self.out_of_boundary_penalty

            # print("Clip Range", clip_low, clip_high)
            action[0:3] = np.clip(action[0:3], clip_low, clip_high)
            action[0:3] = np.clip(action[0:3], -self._eef_displacement, self._eef_displacement)

        self.manipulator_interface.step_action(action)

        if self._done_fn is not None:
            terminal, trunc = self._done_fn(obs)

        return obs, reward, terminal, trunc, {}

    def reset(self, **kwargs) -> tuple:
        self.manipulator_interface.reset(**kwargs)
        obs = self._get_obs()
        return obs, {}

    def obs(self):
        return self._get_obs()

    def _get_obs(self):
        d = {
            'image_primary': self.manipulator_interface.primary_img
        }
        if self._state_encoding == StateEncoding.POS_EULER:
            d['state'] = np.concatenate([
                self.manipulator_interface.eef_pose[:6],
                [0.0],  # padding
                [self.manipulator_interface.gripper_state]], dtype=np.float32
            )
        elif self._state_encoding == StateEncoding.JOINT:
            d['state'] = np.concatenate([
                self.manipulator_interface.joint_states[:7],
                [self.manipulator_interface.gripper_state]], dtype=np.float32
            )

        # Add wrist image to the observation if available
        if self._use_wrist_cam:
            img_wrist = self.manipulator_interface.wrist_img
            if img_wrist is not None:
                # add wrist image to the observation with the desired shape
                d['image_wrist'] = img_wrist
        return d

    def _check_if_within_boundary(self, cartesian_coord: np.ndarray) -> bool:
        """Check if the robot state is within the boundary"""
        if np.any(cartesian_coord < self.workspace_boundary[0, :]):
            return False
        elif np.any(cartesian_coord > self.workspace_boundary[1, :]):
            return False
        return True

##############################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Wiperbot Environment')
    parser.add_argument('--test', action='store_true',
                        help='run the environment test with base class')
    parser.add_argument('--client', action='store_true',
                        help='run the environment with the client')
    parser.add_argument('--widowx_sim', action='store_true',
                        help='run the environment with the simulator')
    parser.add_argument('--viperx', action='store_true',
                        help='run the environment with the viperx')
    parser.add_argument('--widowx', action='store_true',
                        help='run the environment with the widowx')
    parser.add_argument('--host', type=str,
                        default='localhost', help='host of the client')
    parser.add_argument('--log_dir', type=str,
                        default=None, help='log the data to the directory')
    parser.add_argument('--show_img', action='store_true', help='show the image')
    args = parser.parse_args()

    if args.test:
        _interface = ManipulatorInterface()
    elif args.client:
        _interface = ActionClientInterface(host=args.host)
    elif args.widowx_sim:
        # This requires pybullet environment
        from interfaces.widowx_sim import WidowXSimInterface
        _interface = WidowXSimInterface()
    elif args.viperx:
        # This requires rospy environment
        from interfaces.viperx import ViperXInterface
        _interface = ViperXInterface(init_node=True)
    elif args.widowx:
        # This requires rospy environment
        from interfaces.widowx import WidowXInterface
        _interface = WidowXInterface(init_node=True)
    else:
        raise ValueError("Please specify the interface")

    env = ManipulatorEnv(_interface)

    if args.log_dir:
        # this will log the data to the log_dir
        # depends on https://github.com/rail-berkeley/oxe_envlogger
        from oxe_envlogger.envlogger import OXEEnvLogger
        env = OXEEnvLogger(
            env,
            dataset_name="test_mani_env",
            directory=args.log_dir,
            max_episodes_per_file=1,
        )

    # run the environment
    print("init env")

    done = False
    trunc = False
    action = np.array([0.01, 0.00, 0.00, 0.00, 0.0, 0.0, 1.0], dtype=np.float32)

    for eps in range(3):
        obs, _ = env.reset()

        for i in range(20):
            print(f"step {i}")
            obs_tuple = env.step(action)
            obs = obs_tuple[0]
            # print(f"  current proprio: {obs['state']}")

            if args.show_img:
                # convert the image to from bgr to rgb
                rgb = cv2.cvtColor(obs['image_primary'], cv2.COLOR_BGR2RGB)
                cv2.imshow('image', rgb)
                cv2.waitKey(10)
            done, trunc = obs_tuple[2], obs_tuple[3]
            if done or trunc:
                break

    if args.log_dir:
        # close is needed to flush the data
        env.close()

    print("Done logging")
    exit()
