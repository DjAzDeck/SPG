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
For an easy use, we recomment to create a new python3 virtual environment and activate it.
```
python3 -m venv SPG python=3.8
```
Install the requirements 
```
pip3 install -r requirements.txt
```

Train the Agents
----------------
All you have to do is choose the environments and the parameters you wish to test on the `main.py` file and execute the script passing the name argmument `-n`. Example of usage:
```
python main.py -n 1mil --cuda
```

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
```
sudo apt-get install xvfb
```
