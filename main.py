import torch
import numpy as np
import gym
import pybullet_envs
import mujoco_py
import argparse
from multiple_versions import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", required=True, help='Name of the run')
    parser.add_argument("-s", "--seed", required=False, default=0, type=int, help='Seed number for experimentation consistency')
    parser.add_argument("-v", "--version", required=False, default='SPG', type=str, help='Name of the architecture to use')
    parser.add_argument("-p", "--prio", required=False, default=None, help='Switch to True for using prioritize replay buffer')
    parser.add_argument("-t", "--time", required=False, default=1001000, type=int, help='Number of time steps to train the policy')
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Mujoco Environments
    # ENVS = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", 
    #         "Swimmer-v2", "InvertedDoublePendulum-v2", "Reacher-v2"]
    #PyBullet Environments
    ENVS = ['HalfCheetahBulletEnv-v0', 'HumanoidBulletEnv-v0', 'Walker2DBulletEnv-v0', 
            'MinitaurBulletEnv-v0', 'AntBulletEnv-v0', 'HopperBulletEnv-v0']

    VERSIONS = ['DDPG', 'TD3', 'SPG', 'SPGR']
    SEARCHES = [8, 12, 16]
    BATCHES_SIZE = [64, 128, 256]

    TIMESTEPS = args.time

    for env_n in ENVS:
        for vers in VERSIONS:
            for exp_n in SEARCHES:
                for batches in BATCHES_SIZE:
                    #Init environments & SEEDS
                    env = gym.make(env_n)
                    print(env)
                    test_env = gym.make(env_n)
                    env.seed(args.seed)
                    env.action_space.seed(args.seed)
                    torch.manual_seed(args.seed)
                    np.random.seed(args.seed)
                    state_dim = env.observation_space.shape[0]
                    action_dim = env.action_space.shape[0]
                    kwags = {
                        "env": env,
                        "test_env": test_env,
                        "state_dim": state_dim,
                        "action_dim": action_dim,
                        "device": device,
                        "version": vers,
                        "batch_size": batches,
                        "prio": args.prio
                    }
                    print("Benching architecture {} on environment {} ".format(vers, env_n))
                    print("Searching {} times on batch size {} for {} timesteps".format(exp_n, batches, TIMESTEPS))
                    print("Prioritized Replay Buffer: {}".format(args.prio))
                    policy = Trainer(**kwags)
                    name = args.name + '-' + vers + '-' + str(batches) + '-' + str(exp_n) + '-' + str(env_n)
                    policy.train_routine(name, exp_n, TIMESTEPS)