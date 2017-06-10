# Applications of Reinforcement Learning

# Overview

## Robotics and Reinforcement Learning
  - The General Problem of Robotics
  - Applications
    - Toddler
    - What

## Cutting Edge Applications of Reinforcement Learning
  - Poker
  - Multiple Agents
  - Doom

@Li2017

# Robotics and Reinforcement Learning

## The General Problem of Robotics

## Toddler - The Walking Robot

### The Walking Problem
  - Many degrees of freedom causes combinatorial explosion
  - Danger to the robot (damage from falling)
  - Difficult to model physical properties (e.g. friction and pressures on all joints) in order to properly optimize  he robot in simulation
  - Cost to run (can't run robot forever to learn)
  - Delayed reward - "torques applied at one time may have an effect on the performance many steps in the future"

## Toddler - The Walking Robot

### The Robot
 - A simple "passive walker" robot
   - Can walk down a slope just by gravity, i.e. it is a stable platform to learn walking on

#### The Algorithm
 - Uses an Actor-Critic reinforcement learning setup
 - Learns online
 - No world knowledge of the environment

### Results
 - Within one minute, the robot reaches the minimum definition of walking by the researchers:
   - "...foot clearance on nearly every step"
 - Within 20 minutes, it learns a robust gait
   - This equates to around 960 steps (.8 Hz)

##

# The Cutting Edge

## Poker

### Poker as an RL Problem

  - Imperfect information game -- The hands of other players are unknown, as well as the values of upcoming cards
  - Multi-agent Zero Sum game
    -- Nash Equillibrium exists, but is incalculable

### NFSP (Neural Fictitious Self Play)

  - Applying neural networks to the concept of "Fictitious Self Play"
    -- FSP = Choose the best response to the opponent's average behavior
  - Approaches Nash Equilibrium as it learns
  
### NFSP Architecture

  - Remembers state transitions and its best responses in two separate memories Mrl and Msl ***LATEX THIS***
    -- State transitions used in RL; Best responses used for supervised learning
  - ***Mrl*** uses an off-policy deep RL algorithm to learn the best policy from the state transitions
  - ***Msl*** uses a feedforward net to learn the average play (in order to do fictitious self play)
  - Target network for stability and has an explore parameter
  
### NFSP Poker Performance

\columnsbegin
\column{.5\textwidth}

- Comparable to other AIs based on expert knowledge representation (old-AI)
  - Measured in mbb/h -- achieved close to 0 mmb/h
    -- Fold on every hand: -750 mbb/h
    -- Expert: 40-60 mmb/h
    -- Knowledge system based AIs: ~-20 mmb/h

\column{.50\textwidth}

![Performance](/gfx/poker_performance.jpg?raw=true "Poker Performance")
![Nash](/gfx/poker_exploit.jpg?raw=true "Poker Exploitation")

\columnsend
 
@Heinrich2016

## Multiagent RL

### 

## Doom

