from collections import namedtuple
from datetime import datetime
from pathlib import Path
from queue import Queue

import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import get_variables

from .lib import Model, to_tuple, selu


Network = namedtuple('Network', ['y', 'vars', 'ops'])


class DDPG(Model):
    """Deep Deterministic Policy Gradient RL Model."""

    batchsize = 100

    def __init__(self, env_name, memory=1e6, min_memory=1e4,
                 update_frequency=1, state_stacksize=1, checkpoint=None,
                 simulation_workers=2, train_workers=2, feed_workers=1):
        """Create a new DDPG model."""
        super(DDPG, self).__init__(
            env_name=env_name, memory=memory, min_memory=min_memory,
            update_frequency=update_frequency, state_stacksize=state_stacksize,
            simulation_workers=simulation_workers, train_workers=train_workers,
            feed_workers=feed_workers, checkpoint=checkpoint)

    @classmethod
    def make_network(cls, act_states, states, actions, rewards, terminals,
                     states_, training, step, action_bounds):
        """Create the DDPG 4 network network."""
        # Create the online and target actor networks. The online actor
        # once takes the training states and once the 'action states' as
        # inputs and, together with the noise, provides the current action
        with tf.variable_scope('actor'):
            actshape = actions.shape.as_list()[1:]
            actor = cls.make_actor(states, actshape, action_bounds)
            actor_short, _, _ = cls.make_actor(act_states, actshape,
                                               action_bounds, reuse=True)
            action = actor_short + tf.cond(training,
                                           lambda: cls.make_noise(actshape),
                                           lambda: tf.constant(0.))
            action = tf.clip_by_value(action, *action_bounds)  # after noise
            actor_ = cls.make_actor(states_, actshape, action_bounds, 'target')
        tf.contrib.layers.summarize_tensors(actor.vars)

        # Create the online and target critic networks. This has a small
        # speciality: The online critic is created twice, once using the
        # fed states and fed actions as input and once using the fed states
        # and online actor's output as input. The latter is required to compute
        # the `policy gradient` to train the actor. The policy gradient
        # directly depends on how the online policy would currently 'act' in
        # the given state. The important part here is that those two critics
        # (in the following `critic` and `critic_short`) actually are the same
        # network, just with different inputs, but shared (!) parameters.
        with tf.variable_scope('critic'):
            critic = cls.make_critic(states, actions)
            critic_short = cls.make_critic(states, actor.y, reuse=True)
            critic_ = cls.make_critic(states_, actor_.y, 'target')
        tf.contrib.layers.summarize_tensors(critic.vars)

        # Create training and soft update operations.
        train_ops = [
            cls.make_critic_trainer(critic, critic_, terminals, rewards),
            cls.make_actor_trainer(actor, critic_short, step),
            cls.make_soft_updates(critic, critic_),
            cls.make_soft_updates(actor, actor_),
        ]

        # Sync the two network pairs initially.
        init_ops = [cls.make_hard_updates(critic, critic_) +
                    cls.make_hard_updates(actor, actor_)]

        return action, init_ops, train_ops

    @staticmethod
    def dense(x, units, activation=tf.identity, decay=None, minmax=None):
        """Build a dense layer with uniform init and optional weight decay."""
        if minmax is None:
            minmax = float(x.shape[1].value) ** -.5

        return tf.layers.dense(
            x,
            units,
            activation=activation,
            kernel_initializer=tf.random_uniform_initializer(-minmax, minmax),
            bias_initializer=tf.random_uniform_initializer(-minmax, minmax),
            kernel_regularizer=decay and tf.contrib.layers.l2_regularizer(1e-3)
        )

    @classmethod
    def make_critic(cls, states, actions, name='online', reuse=False):
        """Build a critic network q, the value function approximator."""
        with tf.variable_scope(name, reuse=reuse) as scope:
            # training = tf.shape(states)[0] > 1  # Training or evaluating?
            # states = tf.layers.batch_normalization(states, training=training)
            # Feature extraction
            net = cls.dense(states, 100, tf.nn.relu, decay=True)
            # net = tf.layers.batch_normalization(net, training=training)
            net = tf.concat([net, actions], axis=1)  # Actions enter the net
            # Value estimation
            net = cls.dense(net, 50, tf.nn.relu, decay=True)
            y = cls.dense(net, 1, decay=True, minmax=3e-3)
            # ops = get_variables(scope, collection=tf.GraphKeys.UPDATE_OPS)
            return Network(tf.squeeze(y), get_variables(scope), [])

    @staticmethod
    def make_critic_trainer(critic, critic_, terminals, rewards, gamma=.99):
        """Build critic network optimizer minimizing MSE.

        Terminal states are used as final horizon, meaning future rewards are
        only considered if the agent did not reach a terminal state.
        """
        with tf.variable_scope('training/critic'):
            tf.summary.scalar('q/max', tf.reduce_max(critic.y))
            tf.summary.scalar('q/mean', tf.reduce_mean(critic.y))
            targets = tf.where(terminals, rewards, rewards + gamma * critic_.y)
            mse = tf.reduce_mean(tf.squared_difference(targets, critic.y))
            tf.summary.scalar('loss', mse)
            optimizer = tf.train.AdamOptimizer(1e-3)
            with tf.control_dependencies(critic.ops):
                return optimizer.minimize(mse, tf.train.get_global_step())

    @classmethod
    def make_actor(cls, states, dout, bounds, name='online', reuse=False):
        """Build an actor network mu, the policy function approximator."""
        dout = np.prod(dout)
        with tf.variable_scope(name, reuse=reuse) as scope:
            # training = tf.shape(states)[0] > 1  # Training or evaluating?
            # states = tf.layers.batch_normalization(states, training=training)
            net = cls.dense(states, 100, tf.nn.relu)
            # net = tf.layers.batch_normalization(net, training=training)
            net = cls.dense(net, 50, tf.nn.relu)
            y = cls.dense(net, dout, tf.nn.tanh, minmax=3e-3)
            with tf.variable_scope('scaling'):
                olow, ohigh = bounds
                low, high = -1, 1  # fro tanh
                scaled = ((y - low) / (high - low)) * (ohigh - olow) + olow
            # ops = get_variables(scope, collection=tf.GraphKeys.UPDATE_OPS)
            return Network(scaled, get_variables(scope), [])

    @staticmethod
    def make_actor_trainer(actor, critic, step):
        """Build actor network optimizier performing action gradient ascent."""
        with tf.variable_scope('training/actor'):
            # What is `actor.y`'s influence on the critic network's output?
            act_grad, = tf.gradients(critic.y, actor.y)  # (batchsize, dout)
            act_grad = tf.stop_gradient(act_grad)
            # Use `act_grad` as initial value for the `actor.y` gradients --
            # normally this is set to 1s by TF. Results in one value per param.
            policy_gradients = tf.gradients(actor.y, actor.vars, -act_grad)
            mapping = zip(policy_gradients, actor.vars)
            with tf.control_dependencies(actor.ops):
                optimizer = tf.train.AdamOptimizer(1e-4)
                return optimizer.apply_gradients(mapping, global_step=step)

    @staticmethod
    def make_hard_updates(src, dst):
        """Overwrite target with online network parameters."""
        with tf.variable_scope('hardupdates'):
            return [target.assign(online)
                    for online, target in zip(src.vars, dst.vars)]

    @staticmethod
    def make_soft_updates(src, dst, tau=1e-3):
        """Soft update the dst net's parameters using those of the src net."""
        with tf.variable_scope('softupdates'):
            return [target.assign(tau * online + (1 - tau) * target)
                    for online, target in zip(src.vars, dst.vars)]

    @staticmethod
    def make_noise(n, theta=.2, sigma=.4):
        """Ornstein-Uhlenbeck noise process."""
        with tf.variable_scope('OUNoise'):
            shape = to_tuple(n)
            state = tf.Variable(tf.zeros(shape))
            noise = -theta * state + sigma * tf.random_normal(shape)
            # reset = state.assign(tf.zeros((n,)))
            return state.assign_add(noise)
