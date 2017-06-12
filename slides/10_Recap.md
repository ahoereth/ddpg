# Reinforcement Learning Recap

## Problem setting

![RL general setting](./slides/gfx/RL_principle.png){ width=35% }

- An agent is within an environment
- The agent is to complete some task and receive reward
- It solves this task over some amount of time steps

## Relation to Supervised Machine Learning

### In Supervised Learning:

- The environment asks agent a question
- The agent tried to predict the answer
- The environment provides the right answer
- Prediction of agent is compared to the right answer and a loss is computed

### In Reinforcement Learning:

- The environment is in a state, which depends on previous actions of the agent
- The agent takes action
- The agent receives cost from a distribution which is unknown to the agent

Critical: **The agent does not have full access to the function it is trying to
optimize, and therefore must query the function through interaction.**

## Markov Decision Processes

The environment in reinforcement learning can be described as a Markov Decision Process
This relies on the Markov Property (here described as a Markov State):
$\mathbb{P}[S_{t+1}|S_t]=\mathbb{P}[S_{t+1}|S_1...S_t]$
 - This means:
  - The future is independent of the past, given the present
  - The state is a sufficient statistic of the future
  - All previous states can be thrown away and the same result will still be calculated
  
## Markov Decision Processes

Note: For the Markov property to hold, the environment must be fully observable
What does this mean?
- In a *fully observable environment*, the agent's internal state is the same as the environment's internal state
 - i.e., the agent knows how the environment works exactly, and can therefore predict what each of its action will do with 100% accuracy
- In a *partially observale environment*, the agent only indirectly observes the environment's state
 - The agent must construct it's own internal state based on its belief/construction of the environment state
 - Can be represented with a *Partially Observable Markov Decision Process*, POMDP
Put formally, the observation at time $t$ is the same as both the agent's and environment's internal representations
$O_t=S^a_t=S^e_t$

## Markov Decision Processes

Most environments (even if partially observable) can be converted into an MDP

### Going from Chains to Reward Processes

*Chain*: A *Markov Process* or *Chain* is a random sequence of states with the Markov property, defined by tuple:
$\left \langle S,P \right \rangle$
where $S$ is the (finite) state space and $P$ is the state transition matrix (matrix of state transition probabilities)

## Markov Decision Processes

### Going from Chains to Reward Processes

*Markov Reward Process*: Add in reward values to a Markov chain.  Our tuple becomes:
$\left \langle S,P,R,\gamma \right \rangle$
where $R$ is a reward function and $\gamma$ is a discount factor, $\gamma \in [0,1]$
Now that we have reward, we can calculate the total reward of a sequence/chain:
$G_t=R_{t+1}+\gamma R_{t+2}+...=\sum_{k=0}\gamma^k R_{t+k+1}$

## Markov Decision Processes

### Value Functions

The *State-Value Function* gives the long term value of state $s$, i.e. the expected reward if the agent starts in this state
$v(s)=\mathbb{E}(G_t|S_t=s)$


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
