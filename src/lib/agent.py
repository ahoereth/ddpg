from functools import partial
from itertools import count
from threading import Thread

from .environment import Environment
from .memory import Memory


class Agent:
    def __init__(self, env_name, get_action, simulation_queue, *,
                 training=True, memory_size=1e6, min_memory_size=1e4,
                 state_stacksize=1):
        self.env_name = env_name
        self.env = Environment(self.env_name)
        self.get_action = partial(get_action, training=training)
        self.training = training
        self.memory = Memory(state_stacksize=1, capacity=memory_size,
                             observation_shape=self.env.observation_shape,
                             observation_dtype=self.env.observation_dtype,
                             action_shape=self.env.action_shape,
                             action_dtype=self.env.action_dtype)
        self.simulation_queue = simulation_queue
        self.pretrain_steps = min_memory_size
        self.worker = Thread(target=self.simulate, daemon=True)
        self.observation = None
        self.join = self.worker.join
        self.episodes = 0
        self.steps = 0

    def simulate(self, demo=False):
        """Interact with the environment forever."""
        for self.episodes in count(self.episodes):
            self.observation = self.env.reset()
            while not self.env.terminated:
                # Wait until this agent is allowed to move on.
                if not demo and self.pretrain_steps <= 0:
                    self.simulation_queue.get()
                else:
                    self.pretrain_steps -= 1

                # Perform a step in the environment.
                state = self.memory.now(self.observation)
                action = self.get_action(state, training=not demo)
                next_observation, reward, terminal = self.env.step(action)

                # Store the experience in the memory.
                self.memory.add(self.observation, action, reward, terminal)
                self.observation = next_observation

                # Count steps and render if demoing.
                self.steps += 1
                if demo:
                    self.env.render(close=terminal)

            # Stop simulation after one episode when demoing.
            if demo:
                break

    def start(self):
        if not self.worker.is_alive():
            self.worker.start()

    def restart(self):
        self.env.close()
        self.env = Environment(self.env_name)
