**Sample Policy Gradient (SPG)** 
============================

**A complete implementation of deterministic policy gradient algorithms**
--------------------------------
Introduction
----------

This repository contains four determinstic policy gradient algorithm implementations and serves as the backbone of my MSc Thesis research project with title:

**Exploring Deep Reinforcement Learning for Continuous Action Control**

A great amount of work has been put in order to deliver an easy-to-understand and comprehensive implementation of the algorithms. The interested reader should take a look at the complete text (link tba) in order to understand fully the scope of this research.

Setup
--------
For an easy use, we recomment to use `uv` package manager.

```shell
uv venv
uv sync
source .venv/bin/activate
```

Train the Agents
----------------
Training is now driven by a Hydra config (`conf/config.yaml`). Edit that file or override values from the CLI to pick environments, algorithms, hyperparameters, and devices.

Examples:
```bash
# default config
python main.py

# override run name, envs, total steps
python main.py experiment.name=pendulum_run experiment.envs=[Pendulum-v1] experiment.time_steps=20000

# change hyperparameters on the fly
python main.py experiment.prio=false trainer.policy_freq=1 trainer.sigma.start=0.5
```

Each run stores TensorBoard logs and checkpoints under `runs/<run-name>/` (logs in `logs/`, checkpoints in `checkpoints/`). Run names include a timestamp (`YYYYMMDD-HHMMSS`) for quick identification. The resolved Hydra config and applied overrides for each run are saved alongside in `runs/<run-name>/config.yaml` (and `overrides.txt`) for reproducibility.

Environments
------------

The PyBullet environments are already installed from the requirements but for MuJoCo environments, you need to follow the official [Installation Guide](https://github.com/openai/mujoco-py).

Evaluation
----------

In order to evaluate the final agents, you can run the `evaluate_spg.py` script. Example of usage:
```
python evaluate_spg.py -m <weights_path> -r True
```
With the above line you can also record the agent performing on the environments. It requires a virtual graphical display to be installed beforehand.

```shell
sudo apt-get install xvfb
```
