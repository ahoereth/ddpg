from collections import namedtuple
from datetime import datetime
from pathlib import Path
from queue import Queue

import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import get_variables

from .lib import Model, to_tuple, selu


Network = namedtuple('Network', ['y', 'vars', 'ops', 'losses'])


class DDPG(Model):
    """Deep Deterministic Policy Gradient RL Model."""

    batchsize = 100

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
            actor_short = cls.make_actor(act_states, actshape, action_bounds,
                                         reuse=True)
            action = actor_short.y + tf.cond(training,
                                             lambda: cls.make_noise(actshape),
                                             lambda: tf.constant(0.))
            action = tf.clip_by_value(action, *action_bounds)  # after noise
            actor_ = cls.make_actor(states_, actshape, action_bounds,
                                    name='target')
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
    def dense(name, x, units, activation=None, decay=None, minmax=None):
        """Build a dense layer with uniform init and optional weight decay."""
        if minmax is None:
            minmax = 1 / np.sqrt(float(x.shape[1].value))
        initializer = tf.random_uniform_initializer(-minmax, minmax)

        regularizer = None
        if decay is not None:
            regularizer = tf.contrib.layers.l2_regularizer(1e-2)

        return tf.layers.dense(
            x, units,
            name=name,
            activation=activation,
            kernel_initializer=initializer,
            bias_initializer=initializer,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
        )

    @classmethod
    def make_critic(cls, states, actions, name='online', reuse=False):
        """Build a critic network q, the value function approximator."""
        is_batch = tf.shape(states)[0] > 1
        with tf.variable_scope(name, reuse=reuse) as scope:
            states = tf.layers.batch_normalization(states, training=is_batch)
            net = cls.dense('0', states, 100, tf.nn.relu, decay=True)
            net = tf.layers.batch_normalization(net, training=is_batch)
            net = tf.concat([net, actions], axis=1)  # Actions enter the net
            net = cls.dense('1', net, 50, tf.nn.relu, decay=True)
            y = cls.dense('2_q', net, 1, decay=True, minmax=3e-3)
            ops = scope.get_collection(tf.GraphKeys.UPDATE_OPS)
            losses = scope.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            return Network(tf.squeeze(y), get_variables(scope), ops, losses)

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
            loss = tf.losses.mean_squared_error(targets, critic.y)
            tf.summary.scalar('loss', loss)
            if len(critic.losses):
                loss += tf.add_n(critic.losses)
            optimizer = tf.train.AdamOptimizer(1e-3)
            with tf.control_dependencies(critic.ops):
                return optimizer.minimize(loss, tf.train.get_global_step())

    @classmethod
    def make_actor(cls, states, dout, bounds, name='online', reuse=False):
        """Build an actor network mu, the policy function approximator."""
        is_batch = tf.shape(states)[0] > 1
        dout = np.prod(dout)
        with tf.variable_scope(name, reuse=reuse) as scope:
            training = tf.shape(states)[0] > 1
            states = tf.layers.batch_normalization(states, training=is_batch)
            net = cls.dense('0', states, 100, tf.nn.relu)
            net = tf.layers.batch_normalization(net, training=is_batch)
            net = cls.dense('1', net, 50, tf.nn.relu)
            net = tf.layers.batch_normalization(net, training=is_batch)
            y = cls.dense('2', net, dout, tf.nn.tanh, minmax=3e-3)
            scaled = cls.scale(y, bounds_in=(-1, 1), bounds_out=bounds)
            ops = scope.get_collection(tf.GraphKeys.UPDATE_OPS)
            losses = scope.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            return Network(scaled, get_variables(scope), ops, losses)

    @staticmethod
    def make_actor_trainer(actor, critic, step):
        """Build actor network optimizer performing gradient ascent."""
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
    def make_noise(n, theta=.15, sigma=.2):
        """Ornstein-Uhlenbeck noise process."""
        with tf.variable_scope('OUNoise'):
            shape = to_tuple(n)
            state = tf.Variable(tf.zeros(shape))
            noise = -theta * state + sigma * tf.random_normal(shape)
            # reset = state.assign(tf.zeros((n,)))
            return state.assign_add(noise)

    @staticmethod
    def scale(x, bounds_in, bounds_out):
        with tf.variable_scope('scaling'):
            min_in, max_in = bounds_in
            min_out, max_out = bounds_out
            return (((x - min_in) / (max_in - min_in)) *
                    (max_out - min_out) + min_out)
