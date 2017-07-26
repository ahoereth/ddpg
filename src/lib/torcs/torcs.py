import copy
import collections as col
import math
import os
import random
import time


import gym
from gym import spaces
import numpy as np

from .snakeoil3 import Client as snakeoil3

HOST = os.environ.get('TORCS_HOST', 'localhost')
PORT = int(os.environ.get('TORCS_PORT', '3101'))
FILEDIR = os.path.dirname(os.path.realpath(__file__))


class Torcs:
    # Speed limit is applied after this step
    terminal_judge_start = 200

    # [km/h], episode terminates if car is running slower than this limit
    termination_limit_progress = 1

    def __init__(self):
        self.initial_reset = True
        self.initial_run = True
        self.client = None

        # Action Space
        low = [-1., -1.]  # steering, throttle/brake
        high = [1., 1.]  # steering, throttle/brake
        self.action_space = spaces.Box(np.array(low), np.array(high))

        # Observation Space
        low = ([0.] +  # angle
               [0.] * 19 +  # track sensors,
               [-np.inf] +  # trackPos
               [-np.inf, -np.inf, -np.inf] +  # speedX, speedY, speedZ
               [-np.inf] * 4 +  # wheelSpinVel
               [-np.inf])  # rpm
        high = ([1.] +  # angle
                [1.] * 19 +  # track sensors
                [np.inf] +  # trackPos
                [np.inf, np.inf, np.inf] +  # speedX, speedY, speedZ
                [np.inf] * 4 +  # wheelSpinVel
                [np.inf])  # rpm
        self.observation_space = spaces.Box(np.array(low), np.array(high))

    def step(self, action):
        client = self.client

        # Apply Action
        action_torcs = client.R.d
        action_torcs['steer'] = np.clip(action[0], -1, 1)
        throttle = np.clip(np.abs(action[1]), 0, 1)
        if action[1] > 0:
            action_torcs['accel'] = throttle
            action_torcs['brake'] = 0
        else:
            action_torcs['accel'] = 0
            action_torcs['brake'] = throttle

        # Automatic gear shifting
        action_torcs['gear'] = 1
        if client.S.d['speedX'] > 50:
            action_torcs['gear'] = 2
        if client.S.d['speedX'] > 80:
            action_torcs['gear'] = 3
        if client.S.d['speedX'] > 110:
            action_torcs['gear'] = 4
        if client.S.d['speedX'] > 140:
            action_torcs['gear'] = 5
        if client.S.d['speedX'] > 170:
            action_torcs['gear'] = 6

        # Save the previous full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        try:
            client.respond_to_server()  # Apply the Agent's action into torcs
            client.get_servers_input()  # Get the response of TORCS
        except ConnectionError:
            self.initial_reset = True
            raise EnvironmentError('Torcs server went away.')

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an observation from a raw observation vector from TORCS
        self.observation = self.make_observation(obs)

        # Compute reward.
        # TODO: Make plugable
        speed = np.array(obs['speedX'])
        progress = speed * np.cos(obs['angle'])  # forward progress
        reward = speed * (np.cos(obs['angle']) -  # encourage forward
                          np.abs(np.sin(obs['angle'])) -  # discourage sideways
                          np.abs(obs['trackPos']) / 2)  # discourage off-center
        # reward /= 200  # aim for a speed of 200km/h, normalize rewards
        # reward = np.tanh(reward)  # normalize from -1 to 1

        # Collision detection.
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1
            print('terminated due to damage taken')
            episode_terminate = True
            client.R.d['meta'] = True

        # Termination judgement
        episode_terminate = False

        # Episode is terminated if the car is out of track
        if np.min(obs['track']) < 0:
            print('terminated due to out of track')
            reward = -1
            episode_terminate = True
            client.R.d['meta'] = True

        print('progress {:.2f}, reward {:.2f}'.format(progress, reward))

        # Episode terminates if the progress of agent is small
        if self.terminal_judge_start < self.time_step and progress < 1:
            reward = -1
            episode_terminate = True
            client.R.d['meta'] = True
        # if self.terminal_judge_start < self.time_step and speed < 10:
        #     print('terminated due to speed')
        #     episode_terminate = True
        #     client.R.d['meta'] = True

        # Episode is terminated if the agent runs backward
        # if np.cos(obs['angle']) < 0:
        #     print('terminated due to angle')
        #     episode_terminate = True
        #     client.R.d['meta'] = True

        # Send a reset signal
        if client.R.d['meta'] is True:
            self.initial_run = False
            try:
                client.respond_to_server()
            except ConnectionError:
                self.initial_reset = True
                raise EnvironmentError('Torcs server went away.')

        # if episode_terminate:
        #     reward = -1

        self.time_step += 1
        return self.observation, reward, client.R.d['meta'], {}

    def reset(self):
        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            try:
                self.client.respond_to_server()
            except ConnectionError:
                self.initial_reset = True
                return self.reset()

        # Modify here if you use multiple tracks in the environment
        # Open new UDP in vtorcs
        self.client = snakeoil3(H=HOST, p=PORT, vision=False)
        self.client.MAX_STEPS = np.inf

        try:
            self.client.get_servers_input()  # Get the initial input from torcs
        except ConnectionError:
            self.initial_reset = True
            return self.reset()
        obs = self.client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observation(obs)
        self.last_u = None
        self.initial_reset = False
        return self.observation

    def make_observation(self, obs):
        """
        angle, track sensors, trackPos, speedX, speedY, speedZ,
        wheelSpinVel, rpm
        """
        sensors = [('angle', 3.1416),
                   ('track', 200),
                   ('trackPos', 1),
                   ('speedX', 300),
                   ('speedY', 300),
                   ('speedZ', 300),
                   ('wheelSpinVel', 1),
                   ('rpm', 10000)]
        data = [np.array(obs[sensor]) / div for sensor, div in sensors]
        return np.hstack(data)

    def render(self, close=False):
        """Not implemented. Always renders."""
        ...

    def close(self):
        """Shutdown connection to torcs server."""
        self.client.shutdown()
