from datetime import datetime
from pathlib import Path
from queue import Queue

import tensorflow as tf

from .agent import Agent
from .dataflow import Dataflow
from .memory import MultiMemory
from .trainer import Trainer
from .utils import to_tf_dtype


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
        simulation_queue = Queue(max(2, self.update_frequency))

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
        self.training = tf.placeholder_with_default(True, None, 'training')
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
        self.saver = tf.train.Saver(max_to_keep=1)
        if checkpoint:
            self.saver.restore(self.session, checkpoint)
        else:
            self.session.run(tf.global_variables_initializer())

        self.trainers = [Trainer(self.train_step, self.save,
                                 self.training_queue, simulation_queue,
                                 update_frequency=update_frequency)
                         for _ in range(train_workers)]

        for agent in self.agents:
            agent.start()

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

    def save(self, step):
        """Save current model state."""
        if step is None:
            step = self.session.run(self.step)
        self.saver.save(self.session, self.logdir, global_step=step)

    def train_step(self, summarize=False):
        if summarize:
            summary, step, _ = self.session.run([self.summaries, self.step,
                                                 self.train_op],
                                                {self.training: True})
            self.writer.add_summary(summary, step)
        else:
            step, _ = self.session.run([self.step, self.train_op],
                                       {self.training: True})
        return step

    def train(self, steps=1):
        for trainer in self.trainers:
            trainer.start()

        for _ in range(steps):
            self.training_queue.put(1)

        for trainer in self.trainers:
            trainer.join()
