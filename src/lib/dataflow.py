import time
from threading import Thread

import tensorflow as tf
import numpy as np

from .utils import to_tuple


class Dataflow:
    def __init__(self, memory,
                 observation_shape, observation_dtype=tf.float32,
                 action_shape=1, action_dtype=tf.float32,
                 state_stacksize=1, min_memory=None, batchsize=32, workers=1):
        self.memory = memory
        self.batchsize = batchsize
        self.min_memory = batchsize if min_memory is None else min_memory

        state_shape = to_tuple(observation_shape, state_stacksize + 1)
        self.queue = tf.FIFOQueue(
            capacity=self.batchsize * 10,
            dtypes=[observation_dtype, action_dtype, tf.float32, tf.bool],
            shapes=[state_shape, action_shape, [], []]
        )

        # Queue input placeholders.
        self.states = tf.placeholder(observation_dtype,
                                     to_tuple(None, state_shape),
                                     'states')
        self.actions = tf.placeholder(action_dtype,
                                      to_tuple(None, action_shape), 'actions')
        self.rewards = tf.placeholder(tf.float32, (None,), 'rewards')
        self.terminals = tf.placeholder(tf.bool, (None,), 'terminals')

        # self.queue_size = self.queue.size()
        # tf.summary.scalar('misc/queuesize', self.queue_size)
        # tf.summary.histogram('inputs/states', self.states)
        # tf.summary.histogram('inputs/rewards', self.rewards)

        # By default we take samples from the queue, but it is also possible
        # to directly feed them using a `feed_dict`. The latter is for example
        # required when activley using the policy to move in the environment.
        s_stack, a, r, t = self.queue.dequeue_many(self.batchsize)
        s, s_ = tf.squeeze(s_stack[..., :-1]), tf.squeeze(s_stack[..., 1:])
        self.out = (s, a, r, t, s_)

        # This operator will be called in its own thread using the normal
        # feed_dict approach to fill the queue with training samples.
        self.enqueue_op = self.queue.enqueue_many([
            self.states, self.actions, self.rewards, self.terminals,
        ])

        for _ in range(workers):
            Thread(target=self.worker, daemon=True).start()

    def worker(self):
        """Feed the queue with data."""
        while True:  # Feed forever. Enqueue will block when queue is full.
            while len(self.memory) < self.min_memory:
                time.sleep(1)
            batch = self.memory.sample(self.batchsize)
            states, actions, rewards, states_, terminals = zip(*batch)
            self.session.run(self.enqueue_op, {
                self.states: states, self.actions: actions,
                self.rewards: rewards, self.terminals: terminals,
            })
