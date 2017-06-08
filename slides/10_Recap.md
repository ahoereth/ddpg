# RL Recap

## Problem setting


![RL general setting](./slides/gfx/RL_principle.png){ width=35% }

- Lets consider an agent in a stochastic environment
- It sequentially chooses actions over a sequence of time steps -
- such as to maximize a cumulative reward function

## Relation to supervised machine learning
In supervised learning:

- The environment asks agent a question and then provides the right answer
- The agent predicts
- Prediction of agent is compared to the right answer and a loss is computed

In Reinforcement learning:

- The environment is in a state, which depends on previous actions of the agent
- The agent takes action
- The agent receives cost from a distribution which is unknown to the agent

So: **Agent does not have full access to the function she is trying to
optimize, and therefore must query it through interaction.**

## Markov Decision process
Think of this in terms of a Markov decision process (MDP)

This MDP consists of:

- a state space $S$
- an action space $A$
- an initial state distribution with density $p_1(s_1)$
- a stationary transition dynamics distribution with conditional density  $p(s_{t+1}|s_t,a_t)$ which satisfies the Markov property $p(s_{t+1}|s_1,a_1, ..., s_t,a_t) = p(s_{t+1}|s_t,a_t)$ for any trajectory in state-action space $s_1, a_1, s_2, a_2, ... s_T, a_T$
- a reward function $r: S \times A \rightarrow R$


## Policy, Trajectory, Return
- A policy $\pi$ is used to select actions in the MDP
	- Deterministic: $a = \pi(s, \theta)$
	- Stochastic: $\pi(a | s, \theta)$
- This results in a trajectory $\tau$ of states, actions and rewards $s_1, a_1, r_1, ..., s_T, a_T, r_T$  within $S \times A \times \mathbb{R}$
- the return is the total discounted reward from timestep t onwards: $r_t^{\gamma} = \sum_{k=t}^{\infty} \gamma^{k-t} r(s_k,a_k)$ with $0 < \gamma < 1$

## Value Functions
Value functions are expected total discounted reward

- The state-value function is $V^{\pi}(s) = \mathbb{E}[r_1^{\gamma}|S_1 = s; \pi]$
- The state-action value funtion is $Q^{\pi}(s, a) = \mathbb{E}[r_1^{\gamma}|S_1 = s, A_1 = a; \pi]$

The agent's goal is to come up with a policy that maximises the cumulative discounted reward from the start state, $\mathbb{E}[r_1^{\gamma}|\pi_{\theta}]$
