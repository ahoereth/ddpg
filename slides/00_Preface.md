## Deep Reinforcement Learning: Continuous Control

\tableofcontents

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
Value funtions are expected total discounted reward

- The state-value function is $V^{\pi}(s) = \mathbb{E}[r_1^{\gamma}|S_1 = s; \pi]$
- The state-action value funtion is $Q^{\pi}(s, a) = \mathbb{E}[r_1^{\gamma}|S_1 = s, A_1 = a; \pi]$

The agent's goal is to come up with a policy that maximises the cumulative discounted reward from the start state, $\mathbb{E}[r_1^{\gamma}|\pi_{\theta}]$

Policy Gradient Algorithms therfore **adjust the parameters $\theta$ of the policy $\pi$ in the direction of some performance gradient $\nabla_{\theta} \mathbb{E}[r_1^{\gamma}|\pi_{\theta}]$**, in order to maximize expected return 

$$ maximize \mathbb{E}[R | \pi_{\theta} ] $$

## Score Function Gradient Estimator

\begin{align}
\nabla_{\theta} E_x[f(x)] &= \nabla_{\theta} \sum_x p(x) f(x) & \text{definition of expectation} \\
& = \sum_x \nabla_{\theta} p(x) f(x) & \text{swap sum and gradient} \\
& = \sum_x p(x) \frac{\nabla_{\theta} p(x)}{p(x)} f(x) & \text{both multiply and divide by } p(x) \\
& = \sum_x p(x) \nabla_{\theta} \log p(x) f(x) & \text{use the fact that } \nabla_{\theta} \log(z) = \frac{1}{z} \nabla_{\theta} z \\
& = E_x[f(x) \nabla_{\theta} \log p(x) ] & \text{definition of expectation}
\end{align} 

Now we need to sample $x_i \sim\ p(x | \theta)$, and compute $$\hat{g}_i = f(x_i)\nabla_{\theta} log(p(x_i | \theta))$$

## Score function gradient estimator intuition

$$\hat{g}_i = f(x_i)\nabla_{\theta} log(p(x_i | \theta))$$

- $f(x)$ measures how good the sample $x$ is (score function)
- Stepping (ascending) in the direction $\hat{g}_i$ increments the log probability of the $x$, proportionally to the score

## Score function Gradients in context of policies
In the context of policies the random variable x is a whole trajectory $\tau = (s_0 , a_0 , r_0 , s_1 , a_1 , r_1 , ... , s_{T-1} , a_{T-1} , r_{T-1} , s_T )$

$$ \nabla_{\theta} E_{\tau} [R(\tau)] = E_{\tau} [\nabla_{\theta} log \, p(\tau | \theta) R(\tau)] $$

Now we detail $p(\tau | \theta)$:

$$ p(\tau | \theta) = \mu(s_0) \prod_{t=0}^{T-1} [\pi(a_t | s_t , \theta) P(s_{t+1} , r_t | s_t , a_t)] $$

## Score function Gradients in context of policies II

$$ log \, p(\tau | \theta) = log \, \mu(s_0) + \sum_{t=0}^{T-1} [log \, \pi(a_t | s_t , \theta) + log \, P(s_{t+1} , r_t | s_t , a_t)] $$

$$ \nabla_{\theta} log \, p(\tau | \theta) = \nabla_{\theta}  \sum_{t=0}^{T-1} [log \, \pi(a_t | s_t , \theta) $$

$$ \nabla_{\theta} \mathbb{E}_{tau}[R] = \mathbb{E}_{tau}[ R \nabla_{\theta}  \sum_{t=0}^{T-1} [log \, \pi(a_t | s_t , \theta)] $$

