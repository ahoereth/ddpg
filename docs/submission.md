---
title: Continuous Control with Deep Deterministic Policy Gradients
author: John Berroa, Felix Meyer zu Driehausen, Alexander Höreth
date: University of Osnabrück, July 27, 2017
geometry: left=3cm,right=3cm,top=1cm,bottom=2cm
---

Our project can be found on GitHub. In order to ensure that no work after the deadline influences our grade one can refer to the signed commit [915bbb4](https://github.com/ahoereth/ddpg/tree/915bbb4). Besides that we recommend referring to the master branch because we might work on it some more after the deadline:

**[github.com/ahoereth/ddpg](https://github.com/ahoereth/ddpg)**

All following links (except the gifs) link to the master branch.

## Tutorial
The first part of our project is an extensively documented Jupyter Notebook guiding readers through the DDPG algorithm and its implementation and application to a reinforcement learning problem.

**[github.com/ahoereth/ddpg $\rightarrow$ exploration/Lander.ipynb](https://github.com/ahoereth/ddpg/blob/master/exploration/Lander.ipynb)**

Additionally, we provide another notebook providing most basic implementations of different tabular Q-Learning algorithms -- understanding those is a basic requirement before getting into anything deep in the field of reinforcement learning.

**[github.com/ahoereth/ddpg $\rightarrow$ exploration/FrozenLake.ipynb](https://github.com/ahoereth/ddpg/blob/master/exploration/Lander.ipynb)**

## Torcs
Secondly, we applied the DDPG algorithm to the game of torcs -- an old school car racing simulator. Because of the mess that it is to install torcs in a development environment, we used docker for stitching everything together. Getting torcs to run and interfacing with it reasonably was basically a massive chunk of work in itself.

One can view the current state of our results in these two gifs: **[[torcs.gif](https://github.com/ahoereth/ddpg/blob/915bbb4/docs/torcs.gif)]**, **[[torcs2.gif](https://github.com/ahoereth/ddpg/blob/915bbb4/docs/torcs2.gif)]**

Our implementation might also be of some interest engineering wise: We created a library which exposes a single `Model` class which can be inherited from for the implementation of a multitude of deep reinforcement learning algorithms. The class itself provides the interaction with any OpenAI Gym style environment, threaded multi-agent simulation and threaded Tensorflow graph feeding and training. DDPG acts as a first explanatory implementation of that concept:

**[github.com/ahoereth/ddpg $\rightarrow$ src/ddpg.py](https://github.com/ahoereth/ddpg/blob/master/src/ddpg.py)**

Usage explanations and a download link for trained weights can be found in the repository's readme (download in the last paragraph):

**[github.com/ahoereth/ddpg $\rightarrow$ README.md](https://github.com/ahoereth/ddpg/blob/master/src/README.py)**

## Optional Report
One can also read a more detailed report about the theoretical basis behind DDPG and our project if so desired, which is included alongside this document.
