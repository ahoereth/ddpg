from datetime import datetime
from pathlib import Path
from queue import Queue

import tensorflow as tf

from .agent import Agent
from .dataflow import Dataflow
from .memory import MultiMemory
from .trainer import Trainer


class Model:

    batchsize = 32

    def __init__(self, env_name, observation_shape, observation_dtype,
                 action_shape, action_dtype, update_frequency=1,
                 memory=1e6, min_memory=1e4,
                 state_stacksize=1, checkpoint=None,
                 simulation_workers=2, train_workers=2, feed_workers=1,
                 **kwargs):
        self.update_frequency = update_frequency

        time = datetime.now().strftime('%y%m%d-%H%M')
        self.logdir = Path('logs') / type(self).__name__ / time

        tf.train.create_global_step()
        self.step = tf.train.get_global_step()
        self.session = tf.Session()

        # Coordinate multiple simulators with a common memory buffer.
        simulation_queue = Queue(max(2, self.update_frequency))
        agents = [Agent(env_name, self.get_action, simulation_queue,
                        memory_size=memory // simulation_workers,
                        min_memory_size=min_memory // simulation_workers,
                        state_stacksize=state_stacksize)
                  for _ in range(simulation_workers)]
        multi_memory = MultiMemory([agent.memory for agent in agents])

        # Create a single dataflow which feeds samples from the memory buffer
        # to the TensorFlow graph.
        env = agents[0].env
        dataflow = Dataflow(multi_memory,
                            env.observation_shape, env.observation_dtype,
                            env.action_shape, env.action_dtype,
                            batchsize=self.batchsize, workers=feed_workers)

        # TODO(ahoereth): Explain what is going on here
        self.training = tf.placeholder_with_default(True, None, 'training')
        states, actions, rewards, terminals, states_ = dataflow.out
        self.state = tf.placeholder(observation_dtype, observation_shape,
                                    'action_sate')
        action_states = tf.expand_dims(self.state, 0)
        self.action, init_op, train_op = self.make_network(
            action_states, states, actions, rewards, terminals, states_,
            training=self.training, step=self.step, **kwargs)

        # Collect summaries, load checkpoint and/or initialize variables.
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            self.logdir, self.session.graph)
        self.saver = tf.train.Saver(max_to_keep=1)
        if checkpoint:
            self.saver.restore(self.session, checkpoint)
        else:
            self.session.run(tf.global_variables_initializer())

        trainers = [Trainer(self.session, train_op, memory, simulation_queue,
                            self.save, self.step, self.summaries,
                            update_frequency=update_frequency,
                            batchsize=self.batchsize)
                    for _ in range(train_workers)]

        for agent in agents:
            agent.start()

        for trainer in trainers:
            trainer.start()

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
