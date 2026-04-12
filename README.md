# Unitree Go2 Simulation Guide

This guide gives general instructions for getting started with Unitree Go2 simulation using:

- MuJoCo
- NVIDIA Isaac Sim / Isaac Lab

It is intentionally generic and does not assume any particular project layout.

## Overview

A common starting path for Unitree Go2 simulation looks like this:

1. Set up a Linux development environment.
2. Bring up the robot in simulation.
3. Verify standing, teleoperation, and simple velocity commands.
4. Train or load a locomotion policy.
5. Validate behavior in simulation before attempting hardware use.

MuJoCo and Isaac are usually used for different purposes:

- **MuJoCo** is often used for lighter-weight controller testing and sim-to-sim validation.
- **Isaac Sim / Isaac Lab** is often used for GPU-based reinforcement learning and larger-scale training workflows.

## Operating System

### Linux

Linux is the recommended operating system for both MuJoCo and Isaac-based robotics work.

Recommended baseline:

- Ubuntu 22.04 or 24.04
- NVIDIA GPU for Isaac workflows
- Conda or Miniconda

### Windows

If you are using Windows, use **WSL2 with Ubuntu**.

This is the practical default for Linux-oriented robotics tooling. Native PowerShell or Command Prompt should not be treated as the main setup path for these workflows.

Recommended setup:

- Windows 11
- WSL2
- Ubuntu 22.04 or 24.04 inside WSL

Important note:

- For development, package setup, and most shell-based tooling, WSL is the correct path on Windows.
- For Isaac Sim, native Ubuntu is usually the most reliable option for rendering and GPU-heavy simulation.

## Basic Requirements

Install common development tools:

```bash
sudo apt update
sudo apt install -y git cmake build-essential python3 python3-pip
```

Optional but commonly useful:

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

If you prefer Conda:

```bash
conda create -n go2-sim python=3.10 -y
conda activate go2-sim
```

## Getting Started with MuJoCo

## What MuJoCo Is Useful For

MuJoCo is a good choice when you want:

- a lighter simulation stack
- fast iteration on robot controllers
- sim-to-sim testing
- early validation before hardware deployment

## Step 1: Install MuJoCo

Follow the official installation guide:

- https://github.com/google-deepmind/mujoco

Depending on your setup, this may mean:

- installing the Python package
- using a simulator package built on top of MuJoCo
- compiling a robotics simulator that includes MuJoCo-based scenes

## Step 2: Get a Go2-Compatible MuJoCo Simulation Package

You need a simulation package that provides:

- a Go2 robot model
- scene files
- a controller interface
- optional joystick support

Build it according to its instructions. The pattern often looks like:

```bash
git clone <simulation-package>
cd <simulation-package>
mkdir -p build
cd build
cmake ..
make -j
```

## Step 3: Configure the Simulator

Check that the simulator is configured correctly for Go2:

- robot model is set to Go2
- scene file matches the robot
- timestep is reasonable
- controller mode is correct
- joystick or keyboard input is enabled if needed
- communication settings are correct if the controller stack depends on them

## Step 4: Start the Simulator

Typical launch pattern:

```bash
cd <build-directory>
./<simulator-binary>
```

## Step 5: Start the Robot Controller

If there is a separate Go2 controller process, launch it in another terminal:

```bash
cd <controller-build-directory>
./<go2-controller-binary>
```

Typical flow:

1. Start the simulator.
2. Start the controller.
3. Move the robot to a stand or fix-stand state.
4. Enable the locomotion or velocity policy.
5. Test forward, backward, lateral, and turning commands carefully.

## Step 6: Validate Basic Behavior

Before moving further, confirm:

- the robot stands stably
- joint directions are correct
- velocity commands map correctly to motion
- the robot does not immediately drift or collapse
- the controller can tolerate small disturbances

## Getting Started with NVIDIA Isaac Sim / Isaac Lab

## What Isaac Is Useful For

Isaac Sim / Isaac Lab is a good choice when you want:

- reinforcement learning at scale
- vectorized training environments
- GPU acceleration
- more advanced sensor and environment workflows

## Step 1: Install Isaac Lab

Follow the official installation guide:

- https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

Typical requirements:

- Ubuntu
- NVIDIA GPU
- recent NVIDIA driver
- Conda environment

## Step 2: Create and Activate an Environment

Example:

```bash
conda create -n isaac-go2 python=3.10 -y
conda activate isaac-go2
```

Then install Isaac Lab into that environment according to the official instructions.

## Step 3: Add Go2 Robot Assets

A Go2 Isaac workflow usually needs:

- URDF assets
- USD assets
- or a package that already includes the robot definition

Common sources include:

- official Unitree robot descriptions
- Isaac-ready USD assets
- third-party environment packages

Before training, make sure the asset paths are configured correctly.

## Step 4: Verify the Go2 Environment Loads

Before launching training, do a simple smoke test:

- list the available tasks or environments
- load the Go2 scene
- run a preview or play mode if available

The exact commands depend on the training framework you are using.

## Step 5: Train a Go2 Policy

A typical training command in Isaac-style RL workflows looks like:

```bash
python train.py --task <go2-task-name> --headless
```

Before starting a long run, confirm:

- the Go2 task name is correct
- robot assets load without errors
- the GPU is visible
- logs and checkpoints can be written

## Step 6: Play Back the Trained Policy

Typical playback pattern:

```bash
python play.py --task <go2-task-name>
```

Use playback to evaluate:

- gait stability
- turning quality
- command tracking
- recovery after perturbations

## Recommended Learning Order

If you are new to Go2 simulation, this is a practical order:

1. Set up Linux, or Windows with WSL2.
2. Start with MuJoCo for simpler controller bring-up.
3. Verify standing and velocity control.
4. Move to Isaac Sim / Isaac Lab for RL training.
5. Train or load a locomotion policy.
6. Validate the policy in simulation.
7. Only then consider hardware testing.

## Safety Before Real Hardware

Do not move from simulation to the physical robot until you have verified:

- joint ordering
- action scaling
- observation normalization
- startup posture
- emergency stop procedure
- network settings
- safe recovery behavior

A policy that looks stable in simulation can still fail immediately on hardware.

## Troubleshooting

## MuJoCo Starts but the Robot Behaves Wrong

Check:

- the robot model is really Go2
- the scene file matches the model
- joint order matches the controller
- the loaded policy is the intended one

## Isaac Loads but Training Fails

Check:

- asset paths
- Python environment activation
- GPU visibility
- task configuration
- driver compatibility

## Windows Setup Is Unstable

Use WSL2 Ubuntu for development and Linux-based setup steps. For demanding Isaac rendering and GPU workflows, native Ubuntu is often the more reliable choice.

## Suggested README Structure

If you want to turn this into a public GitHub README, use sections like:

1. Overview
2. Supported platforms
3. Windows and WSL note
4. Prerequisites
5. MuJoCo setup
6. Isaac setup
7. Training
8. Playback
9. Validation
10. Hardware safety
