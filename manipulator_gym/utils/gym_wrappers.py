import gym
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import logging
from manipulator_gym.utils.workspace import WorkspaceChecker


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


class LimitMotorMaxEffort(gym.Wrapper):
    """
    Limit the max effort (torque) of the motors to prevent robot damage and failures.
    When the torque limit is reached, do not apply actions.

    NOTE: this currently only works on the widowx interface.

    Args:
    - env: gym environment
    - interface: the interface to the robot
    - torque_limits: the torque limits for each joint. (zhouzypaul: usually effort
        past 1300 is dangerous)
    """

    def __init__(self, env, interface, max_effort_limit=1300):
        super().__init__(env)
        self.interface = interface
        self.max_effort_limit = max_effort_limit
        self.null_action = np.zeros(7)  # widowx specific

    def step(self, action):
        res = self.interface.custom_fn("joint_efforts")
        if max(res.values()) > self.max_effort_limit:
            action = self.null_action
            print_yellow("Warning: Joint effort limit reached. Not applying action.")
            print_yellow(f"Joint efforts: {res}")

        obs, reward, done, trunc, info = self.env.step(action)
        info["joint_efforts"] = res

        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["joint_efforts"] = self.interface.custom_fn("joint_efforts")

        return obs, info


##############################################################################


class CheckAndRebootJoints(gym.Wrapper):
    """
    Every step, check whether joints have failed and reboot them if necessary.
    When joints fail, truncate the episode, and reboot joints on reset.

    NOTE: this currently only works on the widowx interface.

    Args:
    - env: gym environment
    - interface: the interface to the robot
    - check_every_n_steps: check whether the moter ahs failed every n steps, and
        keep track of the failure status. When failed, truncate the episode.
    - force_reboot_per_episode: whether to force reboot all joints on reset.
    """

    def __init__(
        self,
        env,
        interface,
        check_every_n_steps: int = 1,
        force_reboot_per_episode: bool = False,
    ):
        super().__init__(env)
        self.interface = interface
        self.widowx_joints = [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
            "gripper",
        ]
        self.step_number = 0
        self.every_n_steps = check_every_n_steps
        self.force_reboot_per_episode = force_reboot_per_episode
        self.motor_failed = np.zeros_like(self.action_space.sample())

    def step(self, action):
        self.step_number += 1
        res = self.interface.custom_fn("motor_status")

        if res is not None and self.step_number % self.every_n_steps == 0:
            for i, status in enumerate(res):
                if status != 0:
                    self.motor_failed[i] = status
                    break

        obs, reward, done, trunc, info = self.env.step(action)
        if any(self.motor_failed):
            trunc = True

        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        self.step_number = 0

        # reset joints on reset
        res = self.interface.custom_fn("motor_status")
        if res is not None:
            self.env.reset(**{"go_sleep": True})
            for i, status in enumerate(res):
                # soemtime the status here is unreliable, so reboot all joints if previous failure
                if (
                    status != 0
                    or any(self.motor_failed)
                    or self.force_reboot_per_episode
                ):
                    self.interface.custom_fn(
                        "reboot_motor", joint_name=self.widowx_joints[i]
                    )

        self.motor_failed = np.zeros_like(self.action_space.sample())

        null_action = np.zeros(7)
        self.env.step(null_action)  # need to do this so the reset below is not sudden
        return self.env.reset(**kwargs)


##############################################################################


class ConvertState2Proprio(gym.Wrapper):
    """
    convert dict key 'state' to 'proprio' to comply to bridge_dataset stats
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict()
        # convert in obs space
        for key, value in self.env.observation_space.spaces.items():
            if key == "state":
                self.observation_space["proprio"] = value
            else:
                self.observation_space[key] = value

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        obs["proprio"] = obs["state"]
        obs.pop("state")
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["proprio"] = obs["state"]
        obs.pop("state")
        return obs, info


##############################################################################


class ResizeObsImageWrapper(gym.Wrapper):
    """
    resize imags in obs to comply to octo model input
    """

    def __init__(self, env, resize_size: Dict[str, Tuple[int, int]]):
        super().__init__(env)
        self.resize_size = resize_size
        self.observation_space = gym.spaces.Dict()
        # convert in obs space
        for key, value in self.env.observation_space.spaces.items():
            if key in self.resize_size:
                self.observation_space[key] = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(*self.resize_size[key], 3),
                    dtype=np.uint8,
                )
            else:
                self.observation_space[key] = value
        # check if all keys in resize_size are in observation_space
        # else print warning
        for key in self.resize_size:
            if key not in self.env.observation_space.spaces:
                logging.warning(
                    f"Key {key} not in observation_space, ignoring resize for {key}"
                )

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        obs = self._resize_keys_in_obs(obs)
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._resize_keys_in_obs(obs)
        return obs, info

    def _resize_keys_in_obs(self, obs):
        for key in self.resize_size:
            if key in obs:
                obs[key] = cv2.resize(obs[key], self.resize_size[key])
        return obs


##############################################################################


class ClipActionBoxBoundary(gym.Wrapper):
    """
    clip the action to the boundary, ensure ["state"] is provided in obs
    """

    def __init__(
        self,
        env: gym.Env,
        workspace_boundary: np.array,
        rotation_limit: Optional[float] = None,
        out_of_boundary_penalty: float = 0.0,
    ):
        """
        Args:
        - env: gym environment
        - workspace_boundary (2x3 array): the boundary of the eef workspace in abs coordinates
        - rotation_limit: limit the rotation of the eef in radian in [[-rpy], [+rpy]]
        - out_of_boundary_penalty: penalty for going out of the boundary (-ve reward)
           (this should be a negative value.)
        """
        super().__init__(env)
        self._prev_state = None
        self._out_of_boundary_penalty = out_of_boundary_penalty
        self._rotation_limit = rotation_limit
        self.workspace_checker = WorkspaceChecker([workspace_boundary])
        assert (
            "state" in self.env.observation_space.spaces
        ), "state not in observation space"

    def step(self, action):
        """standard gym step function"""
        penalty = 0.0
        if self._prev_state is not None:
            new_point = self._prev_state[0:3] + action[0:3]
            if not self.workspace_checker.within_workspace(new_point):
                print(
                    f"Warning: Action to {new_point} is out of bound. Clipping to workspace boundary."
                )
                penalty = self._out_of_boundary_penalty
                clipped_point = self.workspace_checker.clip_point(new_point)
                action[0:3] = clipped_point - self._prev_state[0:3]

            # Do rotation clipping if limit is provided
            if self._rotation_limit is not None:
                new_rot = self._prev_state[3:6] + action[3:6]

                if np.any(new_rot < self._rotation_limit[0]) or np.any(
                    new_rot > self._rotation_limit[1]
                ):
                    print(
                        "Warning: Rotation out of bounds. Clipping to rotation boundary."
                    )
                    penalty = self._out_of_boundary_penalty
                    clipped_rot = np.clip(
                        new_rot, self._rotation_limit[0], self._rotation_limit[1]
                    )
                    action[3:6] = clipped_rot - self._prev_state[3:6]

        obs, reward, done, trunc, info = self.env.step(action)
        reward -= penalty
        self._prev_state = obs["state"]
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        """standard gym reset function"""
        obs, info = self.env.reset(**kwargs)
        self._prev_state = obs["state"]
        return obs, info

    def visualize_workspace(self, point: Optional[np.array] = None):
        """Util fn to visualize the workspace boundary and optionally
        a point and its clipped version."""
        self.workspace_checker.visualize(point)


##############################################################################


class ClipActionMultiBoxBoundary(ClipActionBoxBoundary):
    """
    User can provide multiple cubloids to define the workspace boundary.

    Action clipping, ensure ["state"] is provided in obs
    """

    def __init__(
        self,
        env: gym.Env,
        cubloids: List[np.array],
        rotation_limit: Optional[float] = None,
        out_of_boundary_penalty: float = 0.0,
    ):
        """
        Args:
        - env: gym environment
        - cubloids (List of 2x3 array): define the workspace boundary of the agent
        - rotation_limit: limit the rotation of the eef in radian in [[-rpy], [+rpy]]
        - out_of_boundary_penalty: penalty for going out of the boundary (-ve reward)
           (this should be a negative value.)
        """
        super().__init__(env)
        self._prev_state = None
        self._out_of_boundary_penalty = out_of_boundary_penalty
        self._rotation_limit = rotation_limit

        self.workspace_checker = WorkspaceChecker(cubloids)
        assert (
            "state" in self.env.observation_space.spaces
        ), "state not in observation space"
