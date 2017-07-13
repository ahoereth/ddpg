from datetime import datetime
from pathlib import Path
from queue import Queue

import tensorflow as tf
import numpy as np

from .agent import Agent
from .dataflow import Dataflow
from .memory import MultiMemory
from .trainer import Trainer
from .utils import to_tf_dtype


def summarize(name, value):
    return tf.Summary(value=[
        tf.Summary.Value(tag=name, simple_value=value)
    ])


class Model:

    batchsize = 32

    def __init__(self, env_name, memory=1e6, min_memory=1e4,
                 update_frequency=1, state_stacksize=1, checkpoint=None,
                 simulation_workers=2, train_workers=2, feed_workers=1):
        self.update_frequency = update_frequency

        time = datetime.now().strftime('%y%m%d-%H%M')
        self.logdir = Path('logs') / type(self).__name__ / time

        self.step = tf.train.create_global_step()
        self.session = tf.Session()

        self.training_queue = Queue(1000)
        simulation_queue = Queue(max(5, self.update_frequency))

        # Coordinate multiple simulators with a common memory buffer.
        self.agents = [Agent(env_name, self.get_action, simulation_queue,
                             memory_size=memory // simulation_workers,
                             min_memory_size=min_memory // simulation_workers,
                             state_stacksize=state_stacksize)
                       for _ in range(simulation_workers)]
        multi_memory = MultiMemory(*[agent.memory for agent in self.agents])

        env = self.agents[0].env
        observation_dtype = to_tf_dtype(env.observation_dtype)
        action_dtype = to_tf_dtype(env.action_dtype)

        # Create a single dataflow which feeds samples from the memory buffer
        # to the TensorFlow graph.
        dataflow = Dataflow(self.session, multi_memory,
                            env.observation_shape, observation_dtype,
                            env.action_shape, action_dtype,
                            state_stacksize=state_stacksize,
                            min_memory=min_memory,
                            batchsize=self.batchsize, workers=feed_workers)

        # TODO(ahoereth): Explain what is going on here
        self.training = tf.placeholder_with_default(True, None, 'is_training')
        states, actions, rewards, terminals, states_ = dataflow.out
        self.state = tf.placeholder(observation_dtype,
                                    env.observation_shape,
                                    'action_state')
        action_states = tf.expand_dims(self.state, 0)
        self.action, init_op, self.train_op = self.make_network(
            action_states, states, actions, rewards, terminals, states_,
            training=self.training, step=self.step,
            action_bounds=env.action_bounds)

        # Collect summaries, load checkpoint and/or initialize variables.
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(str(self.logdir),
                                            self.session.graph)
        self.saver = tf.train.Saver(tf.trainable_variables(),
                                    max_to_keep=1,
                                    keep_checkpoint_every_n_hours=1)
        if checkpoint:
            self.saver.restore(self.session, checkpoint)
        else:
            self.session.run(tf.global_variables_initializer())

        self.trainers = [Trainer(self.train_step, self.save,
                                 self.training_queue, simulation_queue,
                                 update_frequency=update_frequency)
                         for _ in range(train_workers)]

    @classmethod
    def make_network(cls, action_states, states, actions, rewards, terminals,
                     states_, **kwargs):
        """Create the RL network. To be implemented by subclasses."""
        raise NotImplementedError

    def get_action(self, state, training=True):
        """Decide on an action given the current state."""
        action, = self.session.run(self.action, {self.state: state,
                                                 self.training: training})
        return action

    def save(self, step=None):
        """Save current model state."""
        if step is None:
            step = self.session.run(self.step)
        self.saver.save(self.session, str(self.logdir), global_step=step)

    def train_step(self, create_summary=False):
        if summarize:
            summary, step, _ = self.session.run([self.summaries, self.step,
                                                 self.train_op],
                                                {self.training: True})
            self.writer.add_summary(summary, step)
            reward = self.rewards
            self.writer.add_summary(summarize('misc/episodes', self.episodes),
                                    step)
            self.writer.add_summary(summarize('misc/envsteps', self.env_steps),
                                    step)
            self.writer.add_summary(summarize('training/r/avg', reward.mean()),
                                    step)
            self.writer.add_summary(summarize('training/r/max', reward.max()),
                                    step)
        else:
            step, _ = self.session.run([self.step, self.train_op],
                                       {self.training: True})
        return step

    def train(self, steps=1):
        for agent in self.agents:
            if not agent.is_alive():
                agent.start()

        for trainer in self.trainers:
            if not trainer.is_alive():
                trainer.start()

        for _ in range(steps):
            self.training_queue.put(1)

        for trainer in self.trainers:
            trainer.join()

    def demo(self):
        """Demo a single episode."""
        agent = self.agents[0]
        agent.restart()
        agent.simulate(demo=True)  # Not threaded!

    @property
    def env_steps(self):
        """Environment steps taken."""
        return sum([agent.steps for agent in self.agents])

    @property
    def episodes(self):
        """Environment episodes simulated."""
        return sum([agent.episodes for agent in self.agents])

    @property
    def steps(self):
        """Training/SGD steps."""
        return self.session.run(self.steps)

    @property
    def rewards(self):
        rewards = []
        for agent in self.agents:
            rewards.extend(agent.rewards)
        return np.array(rewards)
