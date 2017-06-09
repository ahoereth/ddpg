# Deep Deterministic Policy Gradient

## DQN & DDPG -- Code!

DDPG[^ddpgtutorial] is very simillar to DQN[^dqntutorial] implementation-wise -- just with some added bells and whistles. If you plan to implement DDPG, you might want to start with DQN.

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



## Environments: OpenAI Gym interface

```python
import gym
```

Sensible standardized interface for RL environments. When creating custom environments, building on top of its specifications might make sense.[^gym]

```python
env = gym.make('LunarLanderContinuous-v2')
env.observation_space  # e.g. float vector, 3D array...
env.action_space  # e.g. integer, float vector...
env.reset()
env.step(env.action_space.sample())
env.render()
```

[^gym]: [gym.openai.com](https://gym.openai.com/), [github.com/openai/gym](https://github.com/openai/gym)



## Environment Interaction Loop

Act in the environment following the current policy to generate experiences, store them.

```python
from collections import deque
memory = deque([], maxlen=1e6)  # Note: Random access is O(n)!
policy = lambda state: env.action_space.sample()
done = True
while True:
  if done:
    state = env.reset()
  action = policy(state)
  state_, reward, done, info = env.step(action)
  memory.append((state, action, reward, state_))
  state = state_
```



## The Critic (aka Value Network)

\columnsbegin
\column{.45\textwidth}

**DDPG: $Q(s,a) \rightarrow q$**

**State & action to single Q value**

\column{.45\textwidth}

DQN: $Q(s) \rightarrow \vec q$

State to Q vector, one value per action

\columnsend

```python
def critic(s, a, name):
  with tf.variable_scope(name) as scope:
    net = tf.layers.dense(s, 400, tf.nn.relu)  # Feature detection.
    net = tf.concat([net, a], axis=1)
    net = tf.layers.dense(net, 300, tf.nn.relu)  # Value estimation.
    q = tf.layers.dense(net, 1)  # shape (BATCHSIZE, 1)
    return tf.squeeze(q), get_variables(scope)
qvalues,  thetaQ  = critic(states,  actions, 'online')
qvalues_, thetaQ_ = critic(states_, actions_, 'target')
```



## Training the Critic (Bellman Equation & MSE)

Exactly like in DQN, the critic is optimized to minimize the mean squared error loss between its output and the Bellman approximation. **Note:** No greedy next Q!

\setlength\abovedisplayskip{-1.25em}
\begin{align}
y_t^{DQN} &=r_t + \gamma max_a Q'(s_{t+1},a|\theta^{Q'}) & \text{original DQN target} \\
y_t^{DDPG} &=r_t + \gamma Q'(s_{t+1},\mu'(s_{t+1}|\theta^{\mu'})|\theta^{Q'}) & \text{DDPG critic target} \\
L_t(s_t, a_t) &= \frac{1}{N} \sum^N (Q(s_t, a_t|\theta^{Q}) - y_t^{DDPG})^2 & \text{critic network loss}
\end{align}

```python
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

\column{.45\textwidth}

DQN: $argmax_a Q(s, a) \rightarrow a$

Greedy discrete action selection.

\columnsend


```python
def actor(states, dim_out, name):
  with tf.variable_scope(name) as scope:
    net = dense(states, 400, tf.nn.relu)
    net = dense(net, 300, tf.nn.relu)
    y = dense(net, dim_out, tf.nn.tanh)  # Action scaling.
    return y, get_variables(scope)
actions  = actor(states)
actions_ = actor(states_)
```



## Training the Actor (Policy Gradient Ascent)

```python
action_gradient, = tf.gradients(critic.y, actions)
policy_gradients = tf.gradients(actor.y, actor.vars, -action_gradient)
```


## Target Network Updates

  - Soft Updates
  - tf.assign / tf.Variable.assign for target networks



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

