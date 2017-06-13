# Deep Deterministic Policy Gradient

## DQN & DDPG -- Code!

DDPG[^ddpgtutorial] is very similar to DQN[^dqntutorial] implementation-wise -- just with some added bells and whistles. **If you plan to implement DDPG, you might want to start with DQN.**

  - Define an environment with observations, rewards and actions.
  - Repeatedly act in the environment using the current policy & store experiences.
  - Q network as value function approximator, optimized using the Bellman equation.
  - **New:** Policy network for continuous actions, optimized using policy gradient.
  - Online & target network split. **New:** Soft updates.

```python
import tensorflow as tf
```

[^ddpgtutorial]: [github.com/ahoereth/ddpg $\rightarrow$ DDPG.ipynb](https://github.com/ahoereth/ddpg/blob/master/DDPG-Lander.ipynb) -- **TODO:** update hyperlink
[^dqntutorial]: [github.com/ahoereth/ddpg $\rightarrow$ DQN.ipynb](https://github.com/ahoereth/ddpg) -- **TODO:** not implemented yet



## OpenAI Gym: Environments

![OpenAI Gym Environments [@Brockman2016] [^gym]](gfx/gym.png)

[^gym]: [github.com/openai/gym](https://github.com/openai/gym)



## OpenAI Gym: API

```{.python .numberLines}
import gym
```

Sensible standardized interface for RL environments. When creating custom environments, building on top of its specifications might make sense.

```{.python .numberLines startFrom=2}
env = gym.make('LunarLanderContinuous-v2')
env.observation_space  # e.g. float vector, 3D array...
env.action_space  # e.g. integer, float vector...
env.reset()
action = env.action_space.sample()
state, reward, done, info = env.step(action)
env.render()
```



## Off-Policy Reinforcement Learning: Generating Samples

Act in the environment following the current policy to generate experiences, store them.

```{.python .numberLines}
from collections import deque
memory = deque([], maxlen=1e6)  # Note: Random access is O(n)!
policy = lambda state: env.action_space.sample()
done = True
while True:
  if done:
    state = env.reset()
  action = policy(state)
  state_, reward, done, _ = env.step(action)
  memory.append((state, action, reward, state_))
  state = state_
```



## The Critic (aka Value Network)

\columnsbegin
\column{.45\textwidth}

**DDPG: $Q(s,a) \rightarrow q$**

**State & action to single Q value.**

@Lillicrap2015

\column{.45\textwidth}

DQN: $Q(s) \rightarrow \vec q$

State to Q vector, one value per action.

@Mnih2015

\columnsend

```{.python .numberLines}
def make_critic(states, actions, name):
  with tf.variable_scope(name) as scope:
    net = tf.layers.dense(states, 400, tf.nn.relu)  # Feature extract
    net = tf.concat([net, actions], axis=1)
    net = tf.layers.dense(net, 300, tf.nn.relu)  # Value estimate
    q = tf.layers.dense(net, 1)  # shape (BATCHSIZE, 1)
    return tf.squeeze(q), get_variables(scope)
```



## Training Q-Networks: Bellman Approximation

DQN vs. DDQN vs. DDPG -- fine differences in estimating future reward.

\setlength\abovedisplayskip{-1.25em}

\begin{align}
y^{DQN} &= r_t + \gamma max_a Q'(s_{t+1},a|\theta^{Q'}) & \text{Greedy estimate.} \\
y^{DDQN} &= r_t + \gamma Q'(s_{t+1}, argmax_a Q(s_{t+1}, a)) & \text{Estimate by online policy.} \\
y^{DDPG} &= r_t + \gamma Q'(s_{t+1},\mu'(s_{t+1}|\theta^{\mu'})|\theta^{Q'}) & \text{Estimate by detached policy.}
\end{align}

@Mnih2015, @Hasselt2016, @Lillicrap2015



## Training the DDPG Critic: Bellman Approximation & Mean Squared Error

The critic is optimized to minimize the mean squared error loss between its output and the Bellman approximation.

\setlength\abovedisplayskip{-1.25em}
\begin{align}
y &= r_t + \gamma Q'(s_{t+1},\mu'(s_{t+1}|\theta^{\mu'})|\theta^{Q'}) & \text{Critic Target} \\
\mathbb{L} &= \frac{1}{N} \sum^N (Q(s_t, a_t|\theta^{Q}) - y)^2 & \text{Critic Loss}
\end{align}

```{.python .numberLines}
# critic, _ = make_critic(states, actions, 'online')
# critic_, _ = make_critic(states_, actor_, 'target')
def train_critic(critic, critic_, terminals, rewards):
  targets = tf.where(terminals, rewards, rewards + .99 * critic_)
  mse = tf.reduce_mean(tf.squared(targets - critic))
  return tf.train.AdamOptimizer(1e-3).minimize(mse)
```



## The Actor (aka Policy Network)

\columnsbegin
\column{.45\textwidth}

**DDPG: $\mu(s) \rightarrow a$**

**Vector of continues action values.**

@Lillicrap2015

\column{.45\textwidth}

DQN: $argmax_a Q(s, a) \rightarrow a$

Greedy discrete action selection.

@Mnih2015

\columnsend


```{.python .numberLines}
def make_actor(states, n_actions, name):
  with tf.variable_scope(name) as scope:
    net = dense(states, 400, tf.nn.relu)
    net = dense(net, 300, tf.nn.relu)
    y = dense(net, n_actions, tf.nn.tanh)  # Action scaling.
    return y, get_variables(scope)
```

<!--
actor, thetaMu = make_actor(states, 3, 'online')
actor_, thetaMu_ = make_actor(states_, 3, 'target')
-->



## Training the Actor (Policy Gradient Ascent)

Ascend the gradients of the critic network with respect to the online actor's actions.

\setlength\abovedisplayskip{-1.25em}

\begin{align}
\Delta_{\theta, \mu}J \approx& \Delta_{\theta^\mu} Q(s_t, a|\theta^Q) & a = \mu(s_t|\theta^\mu) \\
=& \Delta_a\ \,Q(s_t, a|\theta^Q) \Delta_{\theta^\mu} a & F'(x) = f'(g(x))g'(x)
\end{align}


```{.python .numberLines}
# actor, thetaMu = make_actor(states, 4, 'online')
# critic, _ = make_critic(states, actor, 'online')
def train_actor(actor, thetaMu, critic):
  value_gradient, = tf.gradients(critic, actor)
  policy_gradients = tf.gradients(actor, thetaMu, -value_gradient)
  mapping = zip(policy_gradients, thetaMu)
  return tf.train.AdamOptimizer(1e-4).apply_gradients(mapping)
```



## Target Network Updates

```{.python .numberLines}
# _, theta = make_critic(states, actions, 'online')
# _, theta_ = make_critic(states_, actor_, 'target')
```

**Hard Updates**: Common in DQN implementations and on initial initialization.

```{.python .numberLines startFrom=3}
def make_hard_update(theta, theta_):
  return [v_.assign(v) for v, v_ in zip(theta, theta_)]
```

**Soft updates**: Slowly follow online parameters, prevents oscillation.

```{.python .numberLines startFrom=5}
def make_soft_update(theta, theta_, tau=1e-3):
  return [v_.assign_sub(tau * v) for v, v_ in zip(theta, theta_)]
```

## Maybe

### Exploration
  - What are different ways of exploration?
  - Balance between exploration/exploitation?
  - Expert Knowledge?

### Ornstein-Uhlenbeck Process
  - In which environments is this great?
  - Gauss-Markov Process
  - Brownian Particles
  - Wiener Process

@Uhlenbeck1930



## Maybe cont.

### Experience Replay

  - Why is it so important for Deep RL?

### Prioritized Experience Replay

  - Why is it better than uniform replay?
  - What are problems one needs to be aware of?
  - Maybe present an implementation?
    - Previously implemented by Alex: Prioritized Replay using a Binary Sum Tree

@Schaul2015



## Maybe cont.

  - Threading? (not implemented as of now)

