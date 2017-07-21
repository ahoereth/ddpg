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
    terminal_judge_start = 500

    # [km/h], episode terminates if car is running slower than this limit
    termination_limit_progress = .5

    initial_reset = True

    def __init__(self, vision=False, throttle=False):
        self.vision = vision
        self.throttle = throttle
        self.initial_run = True
        self.client = None

        self.reset_torcs()

        # Action Space
        low = [-1.]
        high = [1.]
        if throttle is True:
            low += [0.]
            high += [1.]
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
        action = self.agent_to_torcs(action)

        # Apply Action
        action_torcs = client.R.d
        print('action', action)
        print('action_torcs', action_torcs)
        action_torcs['steer'] = np.clip(action['steer'], -1, 1)
        action_torcs['accel'] = np.clip(action['accel'], 0, 1)

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

        client.respond_to_server()  # Apply the Agent's action into torcs
        client.get_servers_input()  # Get the response of TORCS

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an observation from a raw observation vector from TORCS
        self.observation = self.make_observation(obs)

        # Compute reward.
        # TODO: Make plugable
        speed = np.array(obs['speedX'])
        reward = (speed * np.cos(obs['angle']) -
                  np.abs(speed * np.sin(obs['angle'])) -
                  speed * np.abs(obs['trackPos']))
        progress = speed * np.cos(obs['angle'])

        # Collision detection.
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        # Termination judgement
        episode_terminate = False
        # Episode is terminated if the car is out of track
        if np.min(obs['track']) < 0:
            reward = -1
            episode_terminate = True
            client.R.d['meta'] = True

        # Episode terminates if the progress of agent is small
        if self.terminal_judge_start < self.time_step:
            if progress < self.termination_limit_progress:
                episode_terminate = True
                client.R.d['meta'] = True

        # Episode is terminated if the agent runs backward
        if np.cos(obs['angle']) < 0:
            episode_terminate = True
            client.R.d['meta'] = True

        # Send a reset signal
        if client.R.d['meta'] is True:
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1
        return self.observation, reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            if random.random() > .8:
                self.reset_torcs()

        # Modify here if you use multiple tracks in the environment
        # Open new UDP in vtorcs
        self.client = snakeoil3(H=HOST, p=PORT, vision=self.vision)
        self.client.MAX_STEPS = np.inf

        self.client.get_servers_input()  # Get the initial input from torcs
        obs = self.client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observation(obs)
        self.last_u = None
        self.initial_reset = False
        return self.observation

    def end(self):
        # os.system('pkill torcs')
        ...

    def reset_torcs(self):
        # os.system('pkill torcs')
        # time.sleep(0.5)
        # os.system('torcs -nofuel -nolaptime &')
        # time.sleep(0.5)
        # os.system('sh {}'.format(os.path.join(FILEDIR, 'autostart.sh')))
        # time.sleep(0.5)
        ...

    def agent_to_torcs(self, act):
        return {key: val for key, val in zip(['steer', 'accel', 'gear'], act)}

    # def obs_vision_to_image_rgb(self, obs_image_vec):
    #     image_vec = obs_image_vec
    #     rgb = []
    #     temp = []
    #     # convert size 64x64x3 = 12288 to 64x64=4096 2-D list
    #     # with rgb values grouped together.
    #     # Format similar to the observation in openai gym
    #     for i in range(0, 12286, 3):
    #         temp.append(image_vec[i])
    #         temp.append(image_vec[i + 1])
    #         temp.append(image_vec[i + 2])
    #         rgb.append(temp)
    #         temp = []
    #     return np.array(rgb, dtype=np.uint8)

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
