## Applications of Reinforcement Learning

## Overview
# General Applications of Reinforcement Learning
  - What
  - What
# Cutting Edge Applications of Reinforcement Learning
  - What
  - What
  - What
  
 ## General Applications
 
 ## Toddler - The Walking Robot
 # The Walking Problem
  - Many degrees of freedom causes combinatorial explosion
  - Danger to the robot (damage from falling)
  - Difficult to model physical properties (e.g. friction and pressures on all joints) in order to properly optimize the robot in simulation
  - Cost to run (can't run robot forever to learn)
  - Delayed reward - "torques applied at one time may have an effect on the performance many steps in the future"
 # The Robot
  - A simple "passive walker" robot
    - Can walk down a slope just by gravity, i.e. it is a stable platform to learn walking on
 # The Algorithm
  - Uses an Actor-Critic reinforcement learning setup
  - Learns online
  - No world knowledge of the environment
  
  
 # Results
  - Within one minute, the robot reaches the minimum definition of walking by the researchers:
    - "...foot clearance on nearly every step"
  - Within 20 minutes, it learns a robust gait
    - This equates to around 960 steps (.8 Hz)
 
 
 
 Skip skip sip
 
 
 ## The Cutting Edge
 
