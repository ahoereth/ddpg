## Deep Reinforcement Learning: Continuous Control

\tableofcontents


## Index

1. RL Recap

  - Markov stuff
  - Q Learning and why its not enough?
    - Sometimes we need stochastic policies (q is greedy)
    - Discuss silvers example environments which show problems
  - One can optimize the value function (what Q does) or can optimize policy

2. Policy Gradient

  - From true value function f to approximator Q
  - actor critic (from f to Q(theta))

3. Deterministic Policy Gradient

  - From Policy Gradient to Deterministic Policy Gradient

4. Deep Deterministic Policy Gradient

  - Put code on the slides, link the notebook
  - online/target network splits
  - access to variables
  - soft updates
  - tf.gradients
  - fast TF (tf.Queue)?

5. State of the Art: A3C

  - Distributes generating samples
  - Deep RL training bottleneck:
    - Playing the game to generate samples.

6. Applications

  - 2 or 3 applications in-depth
  - focus on real world

