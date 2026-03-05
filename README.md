**Sample Policy Gradient (SPG)** 
============================

**A complete implementation of deterministic policy gradient algorithms**
--------------------------------
Introduction
----------

This repository contains four determinstic policy gradient algorithm implementations and serves as the backbone for the experiments of the paper:

**Sample Policy Gradient: A Competitive Policy Optimisation Method for Off-policy Reinforcement Learning**

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
Training is now driven by a Hydra config (`conf/config.yaml`). Edit that file or override values from the CLI to pick continuous-control environments, algorithms, hyperparameters, and devices.

Examples:
```bash
# default config
python main.py

# quick smoke run on a small continuous task
python main.py experiment.name=pendulum_run 'experiment.envs=[Pendulum-v1]' experiment.time_steps=20000

# run a MuJoCo benchmark
python main.py experiment.name=ant_v5 'experiment.envs=[Ant-v5]' experiment.time_steps=200000

# change hyperparameters on the fly
python main.py experiment.prio=false trainer.policy_freq=1 trainer.sigma.start=0.5 trainer.search.chunk_size=32
```

Each run stores TensorBoard logs and checkpoints under `runs/<run-name>/` (logs in `logs/`, checkpoints in `checkpoints/`). Run names include a timestamp (`YYYYMMDD-HHMMSS`) for quick identification. The resolved Hydra config and applied overrides for each run are saved alongside in `runs/<run-name>/config.yaml` (and `overrides.txt`) for reproducibility.

Environments
------------

Examples of supported environments:
- `Pendulum-v1`
- `MountainCarContinuous-v0`
- `HalfCheetah-v5`
- `Hopper-v5`
- `Walker2d-v5`
- `Ant-v5`
- `Humanoid-v5`


Evaluation
----------

In order to evaluate trained agents, you can run the `evaluate_spg.py` script. It accepts both actor-only weight files and full training checkpoints.

Examples:
```bash
python evaluate_spg.py -m <weights_or_checkpoint_path> -n eval_halfcheetah -e HalfCheetah-v5

python evaluate_spg.py -m <weights_or_checkpoint_path> -n eval_ant -e Ant-v5 -r videos/ant --render-mode rgb_array
```

Use `--render-mode none` for non-interactive evaluation runs. Video recording uses Gymnasium's `RecordVideo` wrapper, and can be ysed with `--render-mode rgb_array`

### Citation

If you like my work, please consider citing me as follows:

*TBA*

### Credits

1. We are using the [ptan](https://github.com/Shmuma/ptan/tree/master) package for fast experience accumulation and replay buffers
