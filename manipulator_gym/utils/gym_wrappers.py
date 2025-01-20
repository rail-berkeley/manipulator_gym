import gym
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import logging
from manipulator_gym.utils.workspace import WorkspaceChecker


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))

def print_red(message):
    print(f"\033[91m{message}\033[0m")  # red color


class TrackTorqueStatus(gym.Wrapper):
    """
    Check the torque status and expose the information in the info dict.

    NOTE: this currently only works on the widowx interface.

    Args:
    - env: gym environment
    """

    def __init__(self, env):
        super().__init__(env)

    def _add_motor_status(self, info):
        res = self.manipulator_interface.custom_fn("get_torque_status")
        info["torque_status"] = res
        return info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        info = self._add_motor_status(info)
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info = self._add_motor_status(info)
        return obs, info


##############################################################################


class LimitMotorMaxEffort(gym.Wrapper):
    """
    Limit the max effort (torque) of the motors to prevent robot damage and failures.
    When the torque limit is reached, do not apply actions.

    NOTE: this currently only works on the widowx interface.

    Args:
    - env: gym environment
    - torque_limits: the torque limits for each joint. (zhouzypaul: usually effort
        past 1300 is dangerous)
    """

    def __init__(self, env, max_effort_limit=1300):
        super().__init__(env)
        self.max_effort_limit = max_effort_limit

    def step(self, action):
        res = self.manipulator_interface.custom_fn("joint_efforts")
        if max(res.values()) > self.max_effort_limit:
            action = np.array([0, 0, 0, 0, 0, 0, action[-1]])  # null action for delta control
            print_yellow("Warning: Joint effort limit reached. Not applying action.")
            print_yellow(f"Joint efforts: {res}")

        obs, reward, done, trunc, info = self.env.step(action)
        info["joint_efforts"] = res

        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["joint_efforts"] = self.manipulator_interface.custom_fn("joint_efforts")

        return obs, info


##############################################################################


class InHouseImpedanceControl(LimitMotorMaxEffort):
    """
    Simple in-house impedance controller for the widowx robots.
    
    Compared to LimitMotorMaxEffort, this not only just clip the actions when the
    joint effort limit is reached, but apply a small action to the reverse direction
    so that the robot doesn't get stuck at the joint effort limit.
    """
    def step(self, action):
        res = self.manipulator_interface.custom_fn("joint_efforts")
        if max(res.values()) > self.max_effort_limit:
            reverse_action = np.concatenate([
                np.array(action)[:6] * -0.5,  # reverse the action
                np.array(action)[-1:]  # keep the gripper action
            ])
            print_yellow("Warning: Joint effort limit reached. Reversing action...")
            action = reverse_action

        obs, reward, done, trunc, info = self.env.step(action)
        info["joint_efforts"] = res

        return obs, reward, done, trunc, info


##############################################################################


class CheckAndRebootJoints(gym.Wrapper):
    """
    Every step, check whether joints have failed and reboot them if necessary.
    There are two types of failure: motor being torqued off and total motor failure.
    If the motor is being torqued off, just torque it back on; if total motor failure,
    we need to reboot the motors (though need to make sure it's in a safe position).

    When joints fail, truncate the episode. On reset, auto torque on / reboot all joints.

    NOTE: this currently only works on the widowx interface.

    Args:
    - env: gym environment
    - check_every_n_steps: check whether the moter ahs failed every n steps, and
        keep track of the failure status. When failed, truncate the episode.
    - force_reboot_per_episode: whether to force reboot all joints on reset.
    """

    def __init__(
        self,
        env,
        check_every_n_steps: int = 1,
        force_reboot_per_episode: bool = False,
    ):
        super().__init__(env)
        self.step_number = 0
        self.every_n_steps = check_every_n_steps
        self.force_reboot_per_episode = force_reboot_per_episode
        self.torque_status = self.manipulator_interface.custom_fn("get_torque_status")
        self.motor_status = self.manipulator_interface.custom_fn("motor_status")

    def step(self, action):
        if self.step_number % self.every_n_steps == 0:
            self.torque_status = self.manipulator_interface.custom_fn(
                "get_torque_status"
            )  # 1 is enabled, 0 is disabled
            self.motor_status = self.manipulator_interface.custom_fn(
                "motor_status"
            )  # 0 is ok, >0 is error code

            # usually the two error codes should imply that the same motors have failed
            assert all(
                (self.motor_status[i] > 0) == (self.torque_status[i] == 0)
                for i in range(len(self.motor_status))
            ), (self.motor_status, self.torque_status)

        obs, reward, done, trunc, info = self.env.step(action)
        if any(self.motor_status) or sum(self.torque_status) < len(self.torque_status):
            trunc = True
        
        self.step_number += 1

        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        self.step_number = 0
        some_joints_failed = any(self.motor_status) or (sum(self.torque_status) < len(self.torque_status))
        
        # if no failures, just reset
        if not some_joints_failed:
            return self.env.reset(**kwargs)
        
        # try to torque on the motors if there's a failure
        if some_joints_failed:
            print_red("Warning: Motor failure detected. Torquing on motors...")
            self.manipulator_interface.custom_fn("enable_torque")
            self.torque_status = self.manipulator_interface.custom_fn(
                "get_torque_status"
            )
        
        # check if torque-on is successful
        if len(self.torque_status) == sum(self.torque_status):
            # successful
            need_to_reboot = self.force_reboot_per_episode
        else:
            # failed
            need_to_reboot = True
        
        # reboot the motors if necessary
        if need_to_reboot:
            print_red("Warning: Motor failure detected. Rebooting motors...")
            assert sum(self.motor_status) > 0  # some motors must have failed
            self.manipulator_interface.custom_fn(
                "safe_reboot_all_motors", go_sleep=kwargs.get("go_sleep", True)
            )  # default moving time 5
            
            # assert that motor status is now ok
            self.motor_status = None
            while self.motor_status is None:
                # need to wait for reboot to finish
                self.motor_status = self.manipulator_interface.custom_fn(
                    "motor_status"
                )
            assert sum(self.motor_status) == 0
            
            # assert that torque status is now ok
            self.torque_status = None
            while self.torque_status is None:
                # need to wait for reboot to finish
                self.torque_status = self.manipulator_interface.custom_fn(
                    "get_torque_status"
                )
            assert sum(self.torque_status) == len(self.torque_status)

        null_action = np.array([0, 0, 0, 0, 0, 0, 1.0])  # TODO: not sure we need this anymore
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
