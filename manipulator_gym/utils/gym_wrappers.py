import gym
import numpy as np
import cv2
from typing import Dict, Tuple
import logging

class ConvertState2Propio(gym.Wrapper):
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
                logging.warning(f"Key {key} not in observation_space, ignoring resize for {key}")

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