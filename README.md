# CSE5100-Fledgling
Getting things off ground.

## About this project
**Fledgling** is a reinforcement learning research project focused on dynamic system identification and control-system design for flying vehicles. The long-term goal is to develop an end-to-end learning-based process that can generate autopilot controllers for known or novel aircraft without relying entirely on traditional manual control-system design.

As a proof of concept, this project begins with a simulated quadrotor and focuses first on a simpler single-axis control problem, likely the **z-axis** for vertical motion such as takeoff and hover. The project explores whether an RL-based controller can learn stable command tracking with performance comparable to, or better than, a human-designed controller.

Rather than solving only one specific control problem, the broader aim is to create a reusable framework for control design that can adapt across different vehicle configurations with minimal changes to actuator and observation definitions.

## Project Goals
The main goals of this project are:

- Design a reinforcement learning architecture that learns vehicle dynamics from observed state feedback
- Train a controller that sends continuous actuator commands in closed loop with a plant model
- Demonstrate stable command following on a simplified proof-of-concept task
- Compare RL controller performance against a deterministic or human-designed baseline controller
- Build toward a generalized workflow for flying vehicles with partially known or unknown dynamics

## Proposed Approach
The project will start with a simulated quadrotor environment and investigate two possible controller-design strategies:

1. **Monolithic model**  
   A single controller that accounts for the full six-degree-of-freedom vehicle dynamics and command-tracking error

2. **Axis-wise / agentic model**  
   Independent learners for decoupled directional axes, followed by a mixer that combines their outputs

The likely first step is the axis-wise approach because it is easier to validate on one axis at a time.

The training process will establish a closed-loop interaction between the controller and plant dynamics. The system will simulate responses to step commands over a fixed time horizon, collect experience trajectories, and optimize a reward based on command-tracking performance and other time-history metrics. If the system enters an unrecoverable state, the invalid portion of the trajectory will be discarded and training will continue with new rollouts.

## Proof-of-Concept Problem
The initial toy problem is expected to be **single-axis command following**, especially vertical motion along the z-axis. A successful early result would be a controller that can perform stable takeoff / hover-like behavior and track commanded altitude changes with acceptable stability and response quality.

Later extensions may include:
- multi-axis motion
- command tracking in multiple directions
- full six-degree-of-freedom simulated flight
- adaptation to other vehicle types beyond a quadrotor

# Getting Started

## Conda Environment
This project uses Conda for environment management.

Create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate fledgling-rl
pip install -r requirements-rl.txt
```