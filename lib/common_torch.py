from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

import ptan

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
def _critic_q(critic: nn.Module, states_v: torch.Tensor, actions_v: torch.Tensor, version: str) -> torch.Tensor:
    if version == "SPGR":
        q1_v, _ = critic(states_v, actions_v)
        return q1_v
    return critic(states_v, actions_v)


@torch.no_grad()
def _vectorized_search_update(
    states_v: torch.Tensor,
    base_actions_v: torch.Tensor,
    critic: nn.Module,
    version: str,
    sigma: float,
    explore: int,
    search_chunk_size: int,
) -> torch.Tensor:
    if explore <= 0:
        return torch.clip(base_actions_v, -1, 1)

    best_actions = base_actions_v.clone()
    best_q = _critic_q(critic, states_v, best_actions, version)
    if best_q.dim() == 1:
        best_q = best_q.unsqueeze(-1)

    noise_std = float(np.sqrt(max(sigma, 1e-12)))
    batch_size = best_actions.shape[0]
    action_dim = best_actions.shape[1]
    state_shape = states_v.shape[1:]
    search_chunk_size = int(search_chunk_size) if search_chunk_size and search_chunk_size > 0 else int(explore)
    remaining = int(explore)

    while remaining > 0:
        chunk = min(remaining, search_chunk_size)
        sampled_actions = best_actions.unsqueeze(0) + (
            torch.randn(chunk, batch_size, action_dim, device=best_actions.device, dtype=best_actions.dtype) * noise_std
        )
        sampled_actions_flat = sampled_actions.view(chunk * batch_size, action_dim)
        repeated_states = (
            states_v.unsqueeze(0)
            .expand(chunk, *states_v.shape)
            .reshape(chunk * batch_size, *state_shape)
        )
        sampled_q = _critic_q(critic, repeated_states, sampled_actions_flat, version)
        if sampled_q.dim() == 1:
            sampled_q = sampled_q.unsqueeze(-1)
        sampled_q = sampled_q.view(chunk, batch_size, 1)
        best_chunk_q, best_chunk_idx = torch.max(sampled_q, dim=0)

        improves = best_chunk_q > best_q
        if torch.any(improves):
            gather_idx = best_chunk_idx.squeeze(-1)
            candidate_best = sampled_actions[gather_idx, torch.arange(batch_size, device=best_actions.device)]
            improve_mask = improves.squeeze(-1)
            best_actions[improve_mask] = candidate_best[improve_mask]
            best_q[improves] = best_chunk_q[improves]
        remaining -= chunk

    return torch.clip(best_actions, -1, 1)


@torch.no_grad()
def unpack_batch_spg(
    batch: Tuple,
    actor: nn.Module,
    critic: nn.Module,
    sigma: float,
    explore: int,
    device: torch.device,
    version: str,
    search_chunk_size: int = 0,
):
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

    elif version in {'SPG', 'SPGR'}:
        policy_actions = actor(states_v)
        q_policy = _critic_q(critic, states_v, policy_actions, version)
        q_replay = _critic_q(critic, states_v, actions_v, version)
        if q_policy.dim() == 1:
            q_policy = q_policy.unsqueeze(-1)
        if q_replay.dim() == 1:
            q_replay = q_replay.unsqueeze(-1)
        choose_replay = q_replay > q_policy
        best_actions = policy_actions.clone()
        replay_mask = choose_replay.expand_as(best_actions)
        best_actions[replay_mask] = actions_v[replay_mask]
        best_action = _vectorized_search_update(
            states_v=states_v,
            base_actions_v=best_actions,
            critic=critic,
            version=version,
            sigma=sigma,
            explore=explore,
            search_chunk_size=search_chunk_size,
        ).to(device)
    else:
        raise ValueError(f"Unsupported SPG unpack version: {version}")

    return states_v, actions_v, rewards_v, dones_t, last_states_v, best_action

def min_Q_value(net, state, action):

    Q1, Q2 = net(state, action)

    return torch.min(Q1, Q2).squeeze()
