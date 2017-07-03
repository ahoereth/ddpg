from functools import partial
from threading import Thread

from .environment import Environment
from .memory import Memory


class Agent:
    def __init__(self, env_name, get_action, simulation_queue, *,
                 training=True, memory_size=1e6, min_memory_size=1e4,
                 state_stacksize=1):
        self.env = Environment(env_name)
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
        self.worker.start()

    def simulate(self):
        """Interact with the environment forever."""
        # Wait until this agent is allowed to move on.
        if self.pretrain_steps == 0:
            self.simulation_queue.get()
        else:
            self.pretrain_steps -= 1

        # Restart environment if it came to an end.
        if self.env.terminated:
            self.observation = self.env.reset()

        # Perform a step in the environment.
        state = self.memory.now(self.observation)
        action = self.get_action(state)
        next_observation, reward, terminal = self.env.step(action)

        # Store the experience in the memory.
        self.memory.add(self.observation, action, reward, terminal)
        self.observation = next_observation

        # Repeat forever.
        self.simulate()
