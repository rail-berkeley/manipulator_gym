import gym
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
import logging


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
                    f"Key {key} not in observation_space, ignoring resize for {key}")

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

    def __init__(self,
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
        self._workspace_boundary = workspace_boundary
        self._prev_state = None
        self._out_of_boundary_penalty = out_of_boundary_penalty
        self._rotation_limit = rotation_limit

        assert "state" in self.env.observation_space.spaces, "state not in observation space"

    def step(self, action):
        """standard gym step function"""
        penalty = 0.0
        if self._prev_state is not None:
            clip_low = self._workspace_boundary[0] - self._prev_state[0:3]
            clip_high = self._workspace_boundary[1] - self._prev_state[0:3]

            if np.any(clip_low > 0) or np.any(clip_high < 0):
                print("Warning: Action out of bounds. Clipping to workspace boundary.")
                penalty = self._out_of_boundary_penalty

            action[0:3] = np.clip(action[0:3], clip_low, clip_high)

            # Check rotation limit if limit is provided
            if self._rotation_limit is not None:
                r_clip_low = self._rotation_limit[0] - self._prev_state[3:6]
                r_clip_high = self._rotation_limit[1] - self._prev_state[3:6]

                if np.any(r_clip_low > 0) or np.any(r_clip_high < 0):
                    print(
                        "Warning: Rotation out of bounds. Clipping to rotation boundary.")
                    penalty = self._out_of_boundary_penalty

                action[3:6] = np.clip(action[3:6], r_clip_low, r_clip_high)

        obs, reward, done, trunc, info = self.env.step(action)
        reward -= penalty
        self._prev_state = obs["state"]
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        """standard gym reset function"""
        obs, info = self.env.reset(**kwargs)
        self._prev_state = obs["state"]
        return obs, info

##############################################################################

class ClipActionMultiBoxBoundary(gym.Wrapper):
    """
    User can provide multiple cubloids to define the workspace boundary.
    
    Action clipping, ensure ["state"] is provided in obs
    """
    def __init__(self,
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
        self._cubloids = cubloids
        self._prev_state = None
        self._out_of_boundary_penalty = out_of_boundary_penalty
        self._rotation_limit = rotation_limit

        assert "state" in self.env.observation_space.spaces, "state not in observation space"

    def step(self, action):
        raise NotImplementedError
