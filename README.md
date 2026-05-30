# VeRT_PPO

An early Proximal Policy Optimization prototype for the VeRT robot at the
Biorobotics Lab, Carnegie Mellon University.

## Overview

This repository is the first stage of the VeRT moving center of mass robot project.
It implements PPO from scratch and trains a control policy inside the CoppeliaSim,
formerly V-REP, simulator. The goal was to validate that a reinforcement learning
policy could control the robot in simulation before moving to a larger and faster
training setup.

The project later moved to NVIDIA Isaac Gym and MuJoCo, added camera based
perception, and grew into the thesis work. That continuation, along with the thesis
itself, lives in
[moving_mass_robot_RL](https://github.com/Yash-Prakash1/moving_mass_robot_RL).

## What it does

* Implements PPO from scratch, including generalized advantage estimation, a clipped
  policy objective, and a separate value loss.
* Defines an MLP actor and critic with a learned log standard deviation for a
  continuous action policy.
* Drives the robot in CoppeliaSim over the ZMQ remote API, sending joint velocity
  commands and reading body pose back each step.
* Logs training to Weights & Biases.

## Stack

* Python, PyTorch
* CoppeliaSim, formerly V-REP, for physics simulation
* The CoppeliaSim ZMQ remote API for control
* Weights & Biases for experiment tracking

## Repository layout

* `ppo1.py`, the PPO training loop
* `RL_net1.py` and `Network.py`, the actor and critic networks
* `vrep_ctrl.py`, the CoppeliaSim interface that loads the scene and steps the robot

## How to run

This project talks to a running CoppeliaSim instance, so you need the simulator
installed and a robot scene open before training.

```bash
# 1. Install CoppeliaSim and open the robot scene
# 2. Make sure the ZMQ remote API is enabled and listening on port 23000
# 3. Install the Python dependencies
pip install torch numpy matplotlib wandb

# 4. Start training
python ppo1.py
```

## Notes

This is an early research prototype. For the more advanced version of the same robot,
with parallel GPU simulation and camera based terrain perception, see
[moving_mass_robot_RL](https://github.com/Yash-Prakash1/moving_mass_robot_RL).
