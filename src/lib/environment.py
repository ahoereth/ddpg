import numpy as np
import gym
from gym import spaces

from .utils import to_tuple


class Environment:
    def __init__(self, env_name):
        self.gym = gym.make(env_name)

        # Determine the environments observation properties.
        observation_space = self.gym.observation_space
        self.is_image = len(to_tuple(self.observation_shape)) == 3
        self.observation_shape = getattr(observation_space, 'shape',
                                         getattr(observation_space, 'n'))
        self.observation_dtype = np.uint8 if self.is_image else np.float

        # Determine the environments action properties.
        action_space = self.gym.action_space
        self.is_discrete = isinstance(action_space, spaces.Discrete)
        self.action_shape = getattr(action_space, 'shape',
                                    getattr(action_space, 'n'))
        self.action_dtype = np.uint8 if self.is_discrete else np.float

        self.terminated = True
        self.observation = None
        self.episode_reward = 0

    def reset(self):
        if self.terminated is True:
            self.episode_reward = 0
            self.observation = self.preprocess(self.gym.reset())
        return self.observation

    def step(self, action):
        observation, reward, terminal, info = self.gym.step(action)
        self.observation = self.preprocess(observation)
        self.episode_reward += reward
        return self.observation, reward, terminal

    def preprocess(self, observation):
        # Extract luminance from RGB images.
        if len(observation.shape) == 3 and observation.shape[2] == 3:
            r, g, b = np.split(observation, 3, axis=-1)
            observation = r // 76 + g // 150 + b // 30  # extract luminance

        # Downsample Atari game observations.
        if shape[0] == 210 and shape[1] == 160:
            observation = observation[34:194:2, 0:160:2, ...]

        return np.squeeze(observation)
