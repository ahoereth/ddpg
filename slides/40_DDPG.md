# Deep Deterministic Policy Gradient

## DQN & DDPG -- Code!

DDPG is very similar to DQN implementation-wise -- just with some added bells and whistles. **If you plan to implement DDPG, you might want to start with DQN.**

  - Define an environment with observations, rewards and actions.
  - Repeatedly act in the environment using the current policy & store experiences.
  - Q network as value function approximator, optimized using the Bellman equation.
  - **New:** Policy network for continuous actions, optimized using policy gradient.
  - Online & target network split. **New:** Soft updates.

```python
import tensorflow as tf
```



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



## Training the Actor (Policy Gradient Ascent)

Ascend the gradients of the critic network with respect to the online actor's actions.

\setlength\abovedisplayskip{-1.25em}

\begin{align}
\Delta_{\theta^\mu}J \approx& \Delta_{\theta^\mu} Q(s_t, a|\theta^Q) & a = \mu(s_t|\theta^\mu) \\
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
  return [dst.assign(src) for src, dst in zip(theta, theta_)]
```

**Soft updates**: Slowly follow online parameters, prevents oscillation.

```{.python .numberLines startFrom=5}
def make_soft_update(theta, theta_, tau=1e-3):
  return [dst.assign(tau * src + (1 - tau) * dst)
          for src, dst in zip(theta, theta_)]
```



## What kind of monster did we just create?

<!--

Nope. No TiKz. Would probably have been prettier.

https://docs.google.com/drawings/d/1D1WgQmXsJNxZn1gmLrsw5XFhftDKE7sAmPxgeNRwVM8

-->

![DDPG Dataflow Graph -- TensorBoard failed us](gfx/DDPG_Structure.png)



## Exploration in Continuous Environments

\columnsbegin

\column{.5\textwidth}

**DQN:** $\epsilon$-greedy -- only for discrete actions.

**DDPG:** Gaussian continuous through time with friction $\theta$ and diffusion $\sigma$ [@Uhlenbeck1930].

\column{.45\textwidth}

![Prototypical Process](gfx/ouprocess.png){height=50%}

\columnsend


```{.python .numberLines}
def noise(n, theta=.15, sigma=.2):
  state = tf.Variable(tf.zeros((n,)))
  noise = -theta * state + sigma * tf.random_normal((n,))
  return state.assign_add(noise)
```


## Do it yourself DDPG

All of the above and more at **[github.com/ahoereth/ddpg $\rightarrow$ Lander.ipynb](https://github.com/ahoereth/ddpg/blob/master/Lander.ipynb)**

- Exhaustively documented. Would recommend if you are interested in Deep RL.
- Critic & actor, online & target networks with soft & hard updates.
- Batch normalization -- disabled because it didn't improve performance.
- Threaded feeding and training:
    - Main thread can focus on generating new experiences.
    - Some threads feed samples from the memory to the TensorFlow graph.
    - Some threads train the network as scheduled by the agent.
- TensorBoard logs with (not so pretty) graph of whats going on.
