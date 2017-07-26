from collections import namedtuple
from datetime import datetime
from functools import partial
from pathlib import Path
from queue import Queue

import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import get_variables

from .lib import Model, to_tuple, to_logpath


Network = namedtuple('Network', ['y', 'vars', 'ops', 'losses'])


class DDPG(Model):
    """Deep Deterministic Policy Gradient RL Model."""

    def __init__(
        self,
        env_name,
        *,
        batchsize=64,
        weight_decay=True,
        bias_decay=True,  # Not defined by original paper.
        decay_scale=1e-2,
        actor_batch_normalization=True,
        critic_batch_normalization=True,
        gamma=0.99,
        critic_learning_rate=1e-3,
        actor_learning_rate=1e-4,
        tau=1e-3,
        mu=0.,
        theta=.15,
        sigma=.2,
        h1=100,
        h2=50,
        config_name='',
        **kwargs
    ):
        self.weight_decay = weight_decay
        self.bias_decay = bias_decay
        self.decay_scale = decay_scale
        self.actor_batch_normalization = actor_batch_normalization
        self.critic_batch_normalization = critic_batch_normalization
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.h1 = h1
        self.h2 = h2
        config_name = to_logpath(
            config_name,
            weightDecay=weight_decay, biasDecay=bias_decay,
            decayScale=decay_scale, actorBN=actor_batch_normalization,
            criticBN=critic_batch_normalization,
            criticLR=critic_learning_rate, actorLR=actor_learning_rate,
            gamma=gamma, tau=tau, env=env_name)
        super(DDPG, self).__init__(env_name, batchsize=batchsize,
                                   config_name=config_name, **kwargs)

    def make_network(self, act_states, states, actions, rewards, terminals,
                     states_, training, action_bounds, exploration_steps):
        """Create the DDPG 4 network network."""
        step = tf.to_float(tf.train.get_global_step())

        # Create the online and target actor networks. The online actor
        # once takes the training states and once the 'action states' as
        # inputs and, together with the noise, provides the current action
        make_noise = partial(self.make_noise, mu=self.mu, theta=self.theta,
                             sigma=self.sigma)
        action_shape = actions.shape.as_list()[1:]
        make_actor = partial(self.make_actor, dout=action_shape,
                             bounds=action_bounds)
        with tf.variable_scope('actor'):
            actor = make_actor(states)
            actor_short = make_actor(act_states, reuse=True)
            actor_ = make_actor(states_, name='target')
            epsilon = tf.maximum(0., (1. - step *
                                      (1. / tf.to_float(exploration_steps))))
            noise = tf.cond(training,
                            lambda: make_noise(action_shape),
                            lambda: tf.constant(0.))
            tf.summary.scalar('misc/epsilon', epsilon)
            tf.summary.histogram('misc/noise', noise)
            action = actor_short.y + epsilon * noise
            action = tf.clip_by_value(action, *action_bounds)  # after noise
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
            critic = self.make_critic(states, actions)
            critic_short = self.make_critic(states, actor.y, reuse=True)
            critic_ = self.make_critic(states_, actor_.y, 'target')
        tf.contrib.layers.summarize_tensors(critic.vars)

        # Create training and soft update operations.
        train_ops = [
            self.make_critic_trainer(critic, critic_, terminals, rewards,
                                     self.gamma, self.critic_learning_rate),
            self.make_actor_trainer(actor, critic_short,
                                    self.actor_learning_rate),
            self.make_soft_updates(critic, critic_, tau=self.tau),
            self.make_soft_updates(actor, actor_, tau=self.tau),
        ]

        # Sync the two network pairs initially.
        init_ops = [self.make_hard_updates(critic, critic_) +
                    self.make_hard_updates(actor, actor_)]

        return action, init_ops, train_ops

    def dense(self, name, x, units, activation=None, decay=None, minmax=None):
        """Build a dense layer with uniform init and optional weight decay."""
        initializer = False
        if minmax is not None:
            # initializer = tf.random_uniform_initializer(-minmax, minmax)
            initializer = tf.random_normal_initializer(0, minmax)
        # else:
        #     fan_in = x.shape[1].value
        #     minmax = 1 / np.sqrt(float(fan_in))
        #     initializer = tf.random_uniform_initializer(-minmax, minmax)

        regularizer = None
        if decay is not None:
            regularizer = tf.contrib.layers.l2_regularizer(self.decay_scale)

        return tf.layers.dense(
            x, units,
            name=name,
            activation=activation,
            kernel_initializer=initializer or None,
            bias_initializer=initializer or tf.zeros_initializer(),
            kernel_regularizer=regularizer if self.weight_decay else None,
            bias_regularizer=regularizer if self.bias_decay else None,
        )

    def make_critic(self, states, actions, name='online', reuse=False):
        """Build a critic network q, the value function approximator."""
        is_batch = tf.shape(states)[0] > 1
        with tf.variable_scope(name, reuse=reuse) as scope:
            net = states
            if self.critic_batch_normalization:
                net = tf.layers.batch_normalization(net, training=is_batch,
                                                    epsilon=1e-7, momentum=.95)
            net = self.dense('0', net, self.h1, tf.nn.relu, decay=True)
            if self.critic_batch_normalization:
                net = tf.layers.batch_normalization(net, training=is_batch,
                                                    epsilon=1e-7, momentum=.95)
            net = tf.concat([net, actions], axis=1)  # Actions enter the net
            net = self.dense('1', net, self.h2, tf.nn.relu, decay=True)
            y = self.dense('2_q', net, 1, decay=True, minmax=1e-4)  # 3e-3)
            ops = scope.get_collection(tf.GraphKeys.UPDATE_OPS)
            losses = scope.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            return Network(tf.squeeze(y), get_variables(scope), ops, losses)

    def make_actor(self, states, dout, bounds, name='online', reuse=False):
        """Build an actor network mu, the policy function approximator."""
        is_batch = tf.shape(states)[0] > 1
        dout = np.prod(dout)
        with tf.variable_scope(name, reuse=reuse) as scope:
            net = states
            if self.actor_batch_normalization:
                net = tf.layers.batch_normalization(net, training=is_batch,
                                                    epsilon=1e-7, momentum=.95)
            net = self.dense('0', net, self.h1, tf.nn.relu)
            if self.actor_batch_normalization:
                net = tf.layers.batch_normalization(net, training=is_batch,
                                                    epsilon=1e-7, momentum=.95)
            net = self.dense('1', net, self.h2, tf.nn.relu)
            if self.actor_batch_normalization:
                net = tf.layers.batch_normalization(net, training=is_batch,
                                                    epsilon=1e-7, momentum=.95)
            y = self.dense('2', net, dout, tf.nn.tanh, minmax=1e-4)  # 3e-3)
            scaled = self.scale(y, bounds_in=(-1, 1), bounds_out=bounds)
            ops = scope.get_collection(tf.GraphKeys.UPDATE_OPS)
            losses = scope.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            return Network(scaled, get_variables(scope), ops, losses)

    @staticmethod
    def make_critic_trainer(critic, critic_, terminals, rewards,
                            gamma=.99, learning_rate=1e-3):
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
            optimizer = tf.train.AdamOptimizer(learning_rate)
            with tf.control_dependencies(critic.ops):
                return optimizer.minimize(loss, tf.train.get_global_step())

    @staticmethod
    def make_actor_trainer(actor, critic, learning_rate=1e-4):
        """Build actor network optimizer performing gradient ascent."""
        with tf.variable_scope('training/actor'):
            # What is `actor.y`'s influence on the critic network's output?
            act_grad, = tf.gradients(critic.y, actor.y)  # (batchsize, dout)
            act_grad = tf.stop_gradient(act_grad)  # TODO: needed?
            # Use `act_grad` as initial value for the `actor.y` gradients --
            # normally this is set to 1s by TF. Results in one value per param.
            policy_gradients = tf.gradients(actor.y, actor.vars, -act_grad)

            # TODO: Investigate the following way to compute the gradient:
            # batchsize = tf.to_float(tf.shape(critic.y)[0])
            # policy_gradients = tf.gradients(critic.y, actor.vars)
            # policy_gradients = [-grad / batchsize
            #                     for grad in policy_gradients]
            mapping = zip(policy_gradients, actor.vars)
            with tf.control_dependencies(actor.ops):
                optimizer = tf.train.AdamOptimizer(learning_rate)
                return optimizer.apply_gradients(mapping,
                                                 tf.train.get_global_step())

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
    def make_noise(n, mu=0., theta=.15, sigma=.2):
        """Ornstein-Uhlenbeck noise process.

        Mu, theta and sigma can either be of size n or floats.
        https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        """
        shape = to_tuple(n)
        with tf.variable_scope('OUNoise'):
            state = tf.get_variable('state', shape,
                                    initializer=tf.constant_initializer(mu))
            noise = theta * (mu - state) + sigma * tf.random_normal(shape)
            # reset = state.assign(tf.zeros((n,)))
            return state.assign_add(noise)

    @staticmethod
    def scale(x, bounds_in, bounds_out):
        min_in, max_in = bounds_in
        min_out, max_out = bounds_out
        with tf.variable_scope('scaling'):
            return (((x - min_in) / (max_in - min_in)) *
                    (max_out - min_out) + min_out)
