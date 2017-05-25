# Deep Deterministic Policy Gradient

## DQN & DDPG

  - Contrast to DQN
    - Also uses Q networks (critic)
    - Q network has only a single output and action is part of its input.
    - Explicit actor
  - Bellman equation approximation

## Deep DPG

  - Custom gradient calculation/application for deep ANNs. Most people probably only use `.minimize(loss)`.

@Lillicrap2015

## Target Networks

  - Soft Updates

## Batch Normalization

## Exploration
  - What are different ways of exploration?
  - Balance between exploration/exploitation?
  - Expert Knowledge?

### Ornstein-Uhlenbeck Process
  - In which environments is this great?
  - Gauss-Markov Process
  - Brownian Particles
  - Wiener Process

@Uhlenbeck1930

## Experience Replay

  - Why is it so important for Deep RL?

### Prioritized Experience Replay

  - Why is it better than uniform replay?
  - What are problems one needs to be aware of?
  - Maybe present an implementation?
    - Previously implemented by Alex: Prioritized Replay using a Binary Sum Tree

@Schaul2015
