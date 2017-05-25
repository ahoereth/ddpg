## Deep Reinforcement Learning: Continuous Control

\tableofcontents


# Deep RL: An Overview
@Li2017

## Experience Replay

  - Why is it so important for Deep RL?

## Prioritized Experience Replay

  - Why is it better than uniform replay?
  - What are problems one needs to be aware of?
  - Maybe present an implementation?
    - Previously implemented by Alex: Prioritized Replay using a Binary Sum Tree

@Schaul2015

## Exploration

  - Why is it important?
  - What are different ways it can be achieved?
  - Balance between exploration/exploitation?
  - How does one need to consider the environment?

## Ornstein-Uhlenbeck Process

  - In which environments is this great?
  - Gauss-Markov Process
  - Brownian Particles
  - Wiener Process

@Uhlenbeck1930

## Another Exploration Strategy in Detail?

  - Specific to a different application would be great.


# Deterministic Policy Gradient

## DPG

  - What is the "policy gradient"? Probably requires some math.

@Silver2014

## Deep DPG

  - Contrast to DQN
    - Also uses Q networks (critic)
    - Q network has only a single output and action is part of its input.
    - Explicit actor
  - Custom gradient calculation/application for deep ANNs. Most people probably only use `.minimize(loss)`.

@Lillicrap2015


# Implementation

## Implementation

  - Walk through the notebook

## Target Networks

    - tf.assign / tf.Variable.assign for target networks

## Tensorflow and custom gradients

    - tf.gradients for action/policy gradient
    - Optimizer.minimize / Optimizer.apply_gradients specifities

```python
action_gradient, = tf.gradients(critic.y, actions)
policy_gradients = tf.gradients(actor.y, actor.vars, -action_gradient)
```


## Moar Tensorflow

    - variable scopes and `get_variables` helpers
    - Batch normalization (`training`?)
    - `tf.control_dependencies`
    - Threading? (not implemented as of now)


# Thank you for your Attention

## References
