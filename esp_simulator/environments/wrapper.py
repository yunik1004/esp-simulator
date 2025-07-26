from typing import List

from gymnasium import spaces
from gymnasium.core import ObservationWrapper, Wrapper
from minigrid.core.constants import OBJECT_TO_IDX


class ObsSelectionWrapper(ObservationWrapper):
    def __init__(self, env, keys: List[str]):
        """A wrapper that makes selected keys as only observations.

        Args:
            env: The environment to apply the wrapper
            keys: List of observation keys
        """
        super().__init__(env)
        self.keys = keys
        self.observation_space = spaces.Dict(
            spaces={key: env.observation_space.spaces[key] for key in self.keys},
        )

    def observation(self, obs):
        return {key: obs[key] for key in self.keys}


class ImgScaleWrapper(ObservationWrapper):
    def __init__(self, env, scale: int = 255 // OBJECT_TO_IDX["agent"]):
        """A wrapper to scale the image

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.scale = scale

    def observation(self, obs):
        obs["image"] = obs["image"] * self.scale
        if "image_full" in obs:
            obs["image_full"] = obs["image_full"] * self.scale
        if "image_cheater" in obs:
            obs["image_cheater"] = obs["image_cheater"] * self.scale
        return obs


class VisualizeFinalStatusWrapper(Wrapper):
    """A wrapper to visualize the final status of the map"""

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.total_reward = 0
        self.total_step = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        self.total_step += 1
        if terminated or truncated:
            info["visual_status"] = self.env.unwrapped.visualize_status()["total"]
            info["total_step"] = self.total_step
            info["total_reward"] = self.total_reward
        return obs, reward, terminated, truncated, info
