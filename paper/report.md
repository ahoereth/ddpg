% Continuous Control with Deep Deterministic Policy Gradients
% John Berroa, Felix Meyer zu Dreyhausen, Alexander Höreth
% University of Osnabrück

# Introduction
With the *Deep Deterministic Policy Gradient*, or DDPG, we want to be able to train an artifical agent through reinforcement learning how to act in an environment in whic the actions it takes are on a continuous scale.  Some examples of this would be car racing (e.g. steering, gas pressure, brake pressuire), flying (throttle, yoke pressures in 2 axes), and games (e.g. Lander).  Continuous control reinforcement learning brings reinforcement learning out of the realm of toy applications and the theoretical and brings it into the continuous world of reality.  To do this however, requires a more complex approach than normal reinforcement learning.  Below we will outline the idea behind DDPG.

## Q-Learning
Reinforcement learning allows an artifical agent to learn a task without being provided with labels clarifying what is the proper action to take in each scenario.  The agent must learn what is proper by getting rewards, and repeating the actions that give higher reward.  The classic reinforcement learning approach, *Q-Learning*, creates a table of State-Action value pairs over many iterations of the agent running through the task, called *episodes*.  This table is then used to pick actions: when in a particular state, the action picks the action that has the highest "Q-value," which if the agent has learned the task properly, will be the most optimal action to take given that state.  Q-learning can be pulled out of the realm of hard caculation and input into an artificial neural network to create a representation of this Q-table.  This is called *Deep Q Network*, or DQN.  DQN has been used in a variety of applications, most notably Atari games.

## Continuous Actions
However, the downfall of DQN and Q-learning is that actions are discrete.  Imagine trying to drive a car, and the goal in a particular state is to avoid a crash.  The wheel must be turned $50^\circ$ (or more) to avoid the obstacle.  A discrete action space such as **Left/Right** might not be enough of a turn to survive.  Discretizing the wheel into $5^\circ$ increments might work, but leads to a combinatorial explosion, as well as not allowing all the degrees of turn in between.  This problem compounds and ultimately leads to the realization that Q-learning is insufficient for continuous actions because of the table representation.  The table would need to be infinitely long in the *Action* axis, and to take the $\textrm{max}$ of an infinite series of numbers is impossible.  On the bright side, continuous actions gives us a policy gradient, and when there's a gradient, there's the ability to optimize.  In this case, maximization is what we want because we want to maximize our future reward.  If we maximize this policy gradient, we will be able to learn continuous actions in reinforcement learning.

The following sections will be laid out as follows: first, we will describe in detail the theoretical basis of DDPG; second, we will discuss how to train DDPG in code with the racing game TORCS, and also how to get the TORCS environment installed; lastly, we will conclude the paper with some comments.

![Q-Table](gfx/continuousq.jpg){width=50%}


# Theoretical Basis

## Policy Gradient

## Deterministic Policy Gradient
Once we have to solved, all we have to do is bring it over into a neural network in order to create the *deep* deterministic policy gradient.

# Implementation

## Tutorial
Because of the complexity of the employed neural network (which is basically four networks), we provide an extensivley documented and interactive tutorial. The tutorial guides the user through a complete implementation of the DDPG algorithm applied to the Lunar Lander (see figure ##) environment from the OpenAI Gym [@Brockman2016].[^lander]

Additionally, targeted for those new to reinforcement learning in general, we provide a notebook all around tabular Q-learning solving the classical Frozen Lake game. In frozen lake one has to control an agent over a small map from start to goal. The challenge is, that the environment is non deterministic: At each step there is a $33%$ chance of slipping and moving orthogonally to the originally executed action.

The notebook implements three algorithms: Q-Learning as taught in the Institute of Cognitive Science's Machine Learning lecture on the deterministic game (meaning there is no slippery ice because the algorithm is unable to solve it otherwise), Q-Learning itself and the classical, non-greedy SARSA algorithm.

The difference between Q-Learning and SARSA in short is that Q-Learning greedily chooses the maximum future Q value for the Bellman equation approximation while SARSA (which fittingly is an abbreviation for *S*tate, *A*ction, *R*eward, *S*tate, *A*ction) uses the current policy for that estimation. For more details, see the notebook.[^lake]

[^lander]: [github.com/ahoereth/ddpg $\rightarrow$ exploration/Lander.ipynb](https://github.com/ahoereth/ddpg/blob/master/exploration/Lander.ipynb)
[^lake]: [github.com/ahoereth/ddpg $\rightarrow$ exploration/FrozenLake.ipynb](https://github.com/ahoereth/ddpg/blob/master/exploration/FrozenLake.ipynb)

## DDPG
- based on a hand full of classes which interact with one another
  - agent & environment & memory, each pair a thread
  - multimemory, enabling sampling from many memories
  - dataflow, using queues for feeding the data to the tensorflow graph
  - trainer, threaded training of a model
  - model, abstract class for implementing reinforcement learning algorithms on gym-like tasks
  - DDPG & DQN (todo), inheriting from the model class and implementing an algorithm each
- only the DDPG class needs to be instantiated with all its hyperparameter settings and the name of an environment, everything else is abstracted


## Torcs
- really old game, hard to get up and running
- abstracted away into a docker container so we can run it remotely


# Training the Deep Deterministic Policy Gradient Algorithm

## The Network

## Obtaining TORCS

# Conclusion
As mentioned in the introduction, continuous action reinforcement learning allows us to do so much more in the real world with the theories behind learning from reward.  The DDPG paper was huge when it came out, and most reinforcement learning papers these days reference it or some derviative work in some way.

