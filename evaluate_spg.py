#!/usr/bin/env python3
import argparse
import gym
import pybullet_envs
import mujoco_py
from tensorboardX import SummaryWriter
from lib import spg_torch

import numpy as np
import torch

#ENV_ID = 'MinitaurBulletEnv-v0'
#ENV_ID = 'HalfCheetahBulletEnv-v0'
#ENV_ID = 'HumanoidBulletEnv-v0'
#ENV_ID = 'Walker2DBulletEnv-v0'
#ENV_ID = 'AntBulletEnv-v0'
ENV_ID = 'HopperBulletEnv-v0'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help='Model file to load')
    parser.add_argument("-e", "--env", default=ENV_ID, help='Environment name to use, default=" + ENV_ID')
    parser.add_argument("-r", "--record", required=False,  help='If specified, sets the recording dir, default=Disabled')
    parser.add_argument("-n", "--name", required=True, help='Name of the test run')
    parser.add_argument("-k", "--kappa", required=False, default=10, help='Number of test runs to estimate the Reward Gain')
    parser.add_argument("-s", "--seed", required=False, default=0, type=int)

    args = parser.parse_args()

    spec = gym.envs.registry.spec(args.env)
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    writer = SummaryWriter(comment="-spg_" + args.name)
    net = spg_torch.SPGActor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(args.model))
    full_reward = []

    for frame_idx in range(args.kappa):
        obs = env.reset()
        total_reward = 0.0
        total_steps = 0
        while True:
            env.render()
            obs_v = torch.FloatTensor([obs])
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            total_steps += 1
            if done:
                break
        print("In %d steps we got %.3f reward" % (total_steps, total_reward))
        full_reward.append(total_reward)
        writer.add_scalar("test_reward", total_reward, frame_idx)
        writer.add_scalar("average_test_reward", np.mean(total_reward), frame_idx)

    print("Reward statistics after 10 runs: \n -------------")
    print("Average reward: %.3f \n \
    Std reward: %.3f \n \
    Max reward: %.3f" % (np.mean(full_reward), np.std(full_reward), np.max(full_reward)))