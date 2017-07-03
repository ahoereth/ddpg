from datetime import datetime
from pathlib import Path
from queue import Queue

import tensorflow as tf

from .agent import Agent
from .trainer import Trainer
from .memory import MultiMemory


class Model:

    batchsize = 32

    def __init__(self, env_name, observation_shape, observation_dtype,
                 action_shape, action_dtype, update_frequency=1,
                 memory=1e6, min_memory=1e4,
                 state_stacksize=1, checkpoint=None,
                 simulation_workers=2, train_workers=2,
                 **kwargs):
        self.update_frequency = update_frequency

        time = datetime.now().strftime('%y%m%d-%H%M')
        self.logdir = Path('logs') / name / time

        tf.train.create_global_step()
        self.step = tf.train.get_global_step()
        self.session = tf.Session()

        # TODO(ahoereth): Explain what is going on here
        self.training = tf.placeholder_with_default(True, None, 'training')
        states, actions, rewards, terminals, states_ = self.dataflow.out
        self.state = tf.placeholder(observation_dtype, observation_shape,
                                    'action_sate')
        action_states = tf.expand_dims(self.state, 0)
        self.action, init_op, train_op = self.make_network(
            action_states, states, actions, rewards, terminals, states_,
            training=training, step=self.step, **kwargs)

        # Collect summaries, load checkpoint and/or initialize variables.
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            self.logdir, self.session.graph)
        self.saver = tf.train.Saver(max_to_keep=1)
        if checkpoint:
            self.saver.restore(self.session, checkpoint)
        else:
            self.session.run(tf.global_variables_initializer())

        # Coordinate multiple simulators and trainers.
        simulation_queue = Queue(max(2, self.update_frequency))
        multi_memory = MultiMemory()
        for _ in range(simulation_workers):
            agent = Agent(env_name, self.get_action, simulation_queue,
                          memory_size=memory // simulation_workers,
                          min_memory_size=min_memory // simulation_workers,
                          state_stacksize=state_stacksize)
            multi_memory.add(agent.memory)
        for _ in range(train_workers):
            trainer = Trainer(self.session, train_op, memory,
                              simulation_queue,
                              update_frequency=update_frequency,
                              batchsize=self.bachsize)

    @classmethod
    def make_network(cls, action_states, states, actions, rewards, terminals,
                     states_, **kwargs):
        """"""
        raise NotImplementedError

    def get_action(self, state, training=True):
        action, = self.session.run(self.action, {self.state: state,
                                                 self.training: training})
        return action

    def worker(self):
        """Train network(s)."""
        while True:  # Train forever. Train steps are limited by agent:
            if task == 'log':
                summary, step, _ = self.session.run([self.summaries,
                                                     self.step,
                                                     self.train_op])
                self.writer.add_summary(summary, step)
            else:
                step, _ = self.session.run([self.step, self.train_op])
            if step % 1000 == 0:  # Save model from time to time.
                self.saver.save(self.session, self.logdir, global_step=step)

            # Every update step allows `update_frequency` environment steps.
            for _ in range(self.update_frequency):
                self.simulation_queue.put(1)  # Blocks if queue is full.
