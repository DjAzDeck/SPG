from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import ptan
from math import sqrt
import random

#N_EXPLORE = 10
#GAMMA = 0.99
#N_STEPS = 1

# def add_sampled_noise(sample, size, sigma, device)->FloatTensor:
# 	noise = 0
# 	chance = random.random()
# 	if chance<=0.5:
#         noise = sample + (torch.randn(size)*torch.sqrt(sigma)).to(device)
#         # noise = sample + np.random.normal(0, sigma, size)
#     else:
#         noise = sample - (torch.randn(size)*torch.sqrt(sigma)).to(device)
# 		# noise = sample - np.random.normal(0, sigma, size)
    
#     return noise

@torch.no_grad()
def unpack_batch(batch: Tuple, device: str = "gpu"):
    """
    Unpack Deep Deterministic Policy Gradient (DDPG) batch
    """
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = ptan.agent.float32_preprocessor(actions).to(device)
    rewards_v = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)

    return states_v, actions_v, rewards_v, dones_t, last_states_v

@torch.no_grad()
def unpack_batch_spg(batch: Tuple, actor: nn.Module, critic: nn.Module, sigma: float, explore: int, device: torch.device, version: str):
    """
    Unpack Sample Policy Gradient (SPG) batch

    Parameters:
        batch: 5-tuple of experience (states, actions, rewards, next_states, dones)
        actor: action selection neural network
        critic: Q-value neural network
        sigma: noise hyperparameter (variance of Gaussian)
        explore: number of searches to be performed
        device: CPU or GPU (cuda)
        version: algorithm that is tested
    Returns: -> Tensors
        states_v: the states of the MDP
        actions_v: the actions of the MDP
        rewards_v: the rewards of the MDP
        dones_t: the finish mask of MDP
        last_states_v: the next state of MDP
        best_action: the best action found by SPG's Offline Gaussian Exploration
    """
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = ptan.agent.float32_preprocessor(actions).to(device)
    rewards_v = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)

    if version == 'SPG_CPU':
        mu_s = actor(states_v)
        a = actions_v.cpu().numpy()
        QcurrentPolicy = critic(states_v, mu_s).cpu().numpy()
        best = mu_s.cpu().numpy()
        QQ = critic(states_v, actions_v).cpu().numpy()
        idxGreater = np.where(QQ > QcurrentPolicy)[0]
        best[idxGreater] = a[idxGreater]
        #Sample new actions around policy 
        for _ in range(explore):
            sampled_np = best + np.random.normal(0, sigma, size=best.shape)
            # sampled_np = add_sampled_noise(best, sigma, best.shape)
            #sampled = ptan.agent.float32_preprocessor(sampled_np).to(device)
            idxGreater = np.where(critic(states_v, ptan.agent.float32_preprocessor(sampled_np).to(device)).cpu().numpy() > \
                                    critic(states_v, ptan.agent.float32_preprocessor(best).to(device)).cpu().numpy() )[0]
            best[idxGreater] = sampled_np[idxGreater]
        best_act = np.clip(best, -1, 1)
        best_action = ptan.agent.float32_preprocessor(best_act).to(device)

    elif version == 'TD3':
        acts_v = actor(states_v)
        q1_v, q2_v = critic(states_v, acts_v)
        # element-wise minimum
        min_Q = torch.min(q1_v, q2_v).squeeze()
        a = actions_v.cpu().numpy()
        best = acts_v.cpu().numpy()
        min_cQ = min_Q_value(critic, states_v, actions_v)
        idxGreater = np.where(min_cQ.cpu().numpy() > min_Q.cpu().numpy())[0]
        best[idxGreater] = a[idxGreater]
        #Sample new actions around policy 
        for _ in range(explore):
            sampled_np = best + np.random.normal(0, sigma, size=best.shape)
            sampled = ptan.agent.float32_preprocessor(sampled_np).to(device)
            best_t = ptan.agent.float32_preprocessor(best).to(device)
            #safe_distr = s_critic(states_v, sampled)
            idxGreater = np.where( (min_Q_value(critic, states_v, sampled).cpu().numpy() > \
                                min_Q_value(critic, states_v, best_t).cpu().numpy()))[0]
            # print(s_critic(states_v, sampled).cpu().numpy())
            best[idxGreater] = sampled_np[idxGreater]
        best_act = np.clip(best, -1, 1)
        best_action = ptan.agent.float32_preprocessor(best_act).to(device)

    elif version == 'SPG':
        mu_s = actor(states_v)
        QcurrentPolicy = critic(states_v, mu_s)
        best = mu_s
        QQ = critic(states_v, actions_v)
        cond_1 = QQ > QcurrentPolicy
        indices_1 = cond_1.nonzero(as_tuple=False)
        best[indices_1] = actions_v[indices_1]
        for _ in range(explore):
            sampled_np = best + (torch.randn(best.size())*sqrt(sigma)).to(device)
            # #Add symmetric noise. (+-)
            # sample_np = add_sampled_noise(best, best.size(), sigma, device)
            Q_1 = critic(states_v, sampled_np)
            Q_2 = critic(states_v, best)
            cond_2 = Q_1 > Q_2
            indices_2 = cond_2.nonzero(as_tuple=False)
            best[indices_2] = sampled_np[indices_2]
        best_action = torch.clip(best, -1, 1).to(device)

    elif version == 'SPGR':
        mu_s = actor(states_v)
        QcurrentPolicy, _ = critic(states_v, mu_s)
        best = mu_s
        QQ, _ = critic(states_v, actions_v)
        cond_1 = QQ > QcurrentPolicy
        indices_1 = cond_1.nonzero(as_tuple=False)
        best[indices_1] = actions_v[indices_1]
        for _ in range(explore):
            sampled_np = best + (torch.randn(best.size())*sqrt(sigma)).to(device)
            # #Add symmetric noise. (+-)
            # sample_np = add_sampled_noise(best, best.size(), sigma, device)
            Q_1, _ = critic(states_v, sampled_np)
            Q_2, _ = critic(states_v, best)
            cond_2 = Q_1 > Q_2
            indices_2 = cond_2.nonzero(as_tuple=False)
            best[indices_2] = sampled_np[indices_2]
        best_action = torch.clip(best, -1, 1).to(device)

    return states_v, actions_v, rewards_v, dones_t, last_states_v, best_action

def min_Q_value(net, state, action):

    Q1, Q2 = net(state, action)

    return torch.min(Q1, Q2).squeeze()