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

*Markov Reward Process*: Add in reward values to a Markov chain.  Our tuple becomes:
$\left \langle S,P,R,\gamma \right \rangle$
where $R$ is a reward function and $\gamma$ is a discount factor, $\gamma \in [0,1]$
Now that we have reward, we can calculate the total reward of a sequence/chain:
$G_t=R_{t+1}+\gamma R_{t+2}+...=\sum_{k=0}\gamma^k R_{t+k+1}$

## Markov Decision Processes

### Value Functions

The *State-Value Function* gives the long term value of state $s$, i.e. the expected reward if the agent starts in this state
$V(s)=\mathbb{E}(G_t|S_t=s)$
We have to take the expectation because $G_t$ is random; we need to know the expected value based on all random permutations of traversals through the Markov process

## Markov Decision Processes

### Now to the Markov Decision Process

- Up until now, actions have been completely random
- Now we add a policy to choose actions
It can be described as an "MRP with decisions."  We add in the agent to our tuple representation:
$\left \langle S,A,P,R,\gamma \right \rangle$
where $A$ is a finite set of actions
$P$ and $R$ now depend on actions taken; formally:
$P_{ss'}^a=\mathbb{P}(R_{t+1}|S_t=s, A=a)$
$R_{s}^a=\mathbb{E}(R_{t+1}|S_t=s, A=a)$

Reminder: A *policy* is a function mapping states to actions; it tells the agent what to do given the state

## Markov Decision Processes

### Value Functions Revisited

The *State-Value Function* remains mostly the same in MDP as in MRP, except it depends on the policy:
$V^\pi(s)=\mathbb{E_\pi}(G_t|S_t)$
"The expectation when we sample all actions according to this policy $\pi$"; the value of a state
The *Action-Value Function* is defined as how good it is to take a particular action when the agent is in a particular state:
$Q^\pi(s,a)=\mathbb{E_\pi}(G_t|S_t=s, A_t=a)$
"The expected return starting from state $s$, taking action $a$, and then following policy $\pi$"; the value of an action

Note: we can also define these as recursive *Bellman Equations* where they refer to themselves instead of with $G_t$

## Markov Decision Processes

### Solving Reinforcement Learning
- We want to maximize the value of our actions based on future reward, therefore (* denotes max/optimal function):
$V^*(s)=\max_a Q^*(s,a)$
$Q^*(s,a)=R_s^a+\gamma \sum_{s'\in S} P_{ss'}^a V^*(s')$
We can nest these to get:
$Q^*(s,a)=R_s^a+\gamma \sum_{s'\in S} P_{ss'}^a \max_a Q^*(s,a)$
This is the *Bellman Optimality Equation* (note: it can be nested in the other direction too to solve for $v_*(s)$)
Solve this, and the reinforcement learning problem is solved


	- Deterministic: $a = \pi(s, \theta)$
	- Stochastic: $\pi(a | s, \theta)$

