#!/usr/bin/env python3
import os
import ptan
import time
import copy
import gym
from tensorboardX import SummaryWriter
import numpy as np
from lib import  spg_torch, common_torch
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


SIGMA_DECAY_LAST_FRAME = 1200000
SIGMA_START = 0.7
SIGMA_FINAL = 0.01

PRIO_REPLAY_ALPHA = 0.6
beta_start = 0.4
beta_frames = 400000
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

def test_net(net: nn.Module, env: gym.Env, count: int = 10, device: str = "cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_s = net(obs_v)
            action = mu_s.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


class Trainer(object):
    def __init__(
        self,
        env: gym.Env,
        test_env: gym.Env,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        version: str,
        batch_size: int,
        prio: bool = None,
        replay_size: int = 1000000,
        discount: float = 0.99,
        LR: float = 1e-4,
        prio_alpha: float = 0.6,
        rollout_step: int = 1,
        policy_freq: int = 2,
        noise_clip: float = 0.5,
        policy_noise: float = 0.2,
        alpha: float = 2.5,
        tau: float = 0.005
        ):
        
        self.LR = LR
        self.device = device
        self.prio = prio
        self.prio_alpha = prio_alpha
        self.batch_size = batch_size
        self.gamma = discount
        self.roll_steps = rollout_step
        self.test_env = test_env
        self.replay_initial = 20000
        self.test_iters = 5000
        self.version = version
        self.policy_freq = policy_freq
        self.policy_noise = policy_noise
        self.alpha = alpha
        self.tau = tau
        self.noise_clip = noise_clip

        #Init nets
        if self.version == 'DDPG':
            self.actor = spg_torch.SPGActor(state_dim, action_dim).to(self.device)
            self.actor_tgt = ptan.agent.TargetNet(self.actor)
            self.critic = spg_torch.SPGCritic(state_dim, action_dim).to(self.device)
            self.critic_tg = ptan.agent.TargetNet(self.critic)
            self.agent = spg_torch.AgentDDPG(self.actor, device=self.device)
        elif self.version == 'SPG':
            self.actor = spg_torch.SPGActor(state_dim, action_dim).to(device)
            self.agent = spg_torch.AgentSPG(self.actor, device=self.device)
            self.actor_tgt = ptan.agent.TargetNet(self.actor)
            self.critic = spg_torch.SPGCritic(state_dim, action_dim).to(device)
            self.critic_tg = ptan.agent.TargetNet(self.critic)
        elif (self.version == 'TD3') or (self.version == 'SPGR'):
            self.actor = spg_torch.SPGActor(state_dim, action_dim).to(device)
            self.agent = spg_torch.AgentSPG(self.actor, device=self.device)
            self.actor_tgt = copy.deepcopy(self.actor)
            self.critic = spg_torch.Critic(state_dim, action_dim).to(device)
            self.critic_tg = copy.deepcopy(self.critic)
        print(self.actor)
        print(self.critic)

        #Init optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.LR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.LR)
        #Init experience source & replay buffer
        self.experience_source = ptan.experience.ExperienceSourceFirstLast(
        env, self.agent, gamma=discount, steps_count=1)
        if self.prio:
            self.buffer = ptan.experience.PrioritizedReplayBuffer(
            self.experience_source, buffer_size=replay_size, alpha=PRIO_REPLAY_ALPHA)
        else:
            self.buffer = ptan.experience.ExperienceReplayBuffer(
            self.experience_source, buffer_size=replay_size)


    def train_routine(self, name: str, explore: int, stop: int):
        #Make save path
        self.save_path = os.path.join("saves", name)
        os.makedirs(self.save_path, exist_ok=True)
        #init writer for logging
        self.writer = SummaryWriter(comment='-' + name)
        self.frame_idx = 0
        self.best_reward = None
        with ptan.common.utils.RewardTracker(self.writer) as tracker:
            with ptan.common.utils.TBMeanTracker(
                    self.writer, batch_size=10) as tb_tracker:
                while True:
                    self.frame_idx += 1
                    #Populate the bufffer with samples-trajectories
                    self.buffer.populate(1)
                    rewards_steps = self.experience_source.pop_rewards_steps()
                    if rewards_steps:
                        rewards, steps = zip(*rewards_steps)
                        tb_tracker.track("episode_steps", steps[0], self.frame_idx)
                        tracker.reward(rewards[0], self.frame_idx)

                    if len(self.buffer) < self.replay_initial:
                        continue
                    #Decrease sigma during training
                    sigma = max(SIGMA_FINAL, SIGMA_START - \
                    self.frame_idx / SIGMA_DECAY_LAST_FRAME)
                    sigma = round(sigma, 3)

                    if self.prio:
                        beta = beta_by_frame(self.frame_idx)
                        batch, batch_indices, batch_weights = self.buffer.sample(self.batch_size, beta)
                        batch_weights_v = torch.tensor(batch_weights).to(self.device)
                    else:
                        batch = self.buffer.sample(self.batch_size)
                    
                    if (self.version == 'SPG') or (self.version == 'SPGR'): 
                        train_states_v, train_actions_v, train_rewards_v, \
                        train_dones_mask, train_last_states_v, best_actions = \
                            common_torch.unpack_batch_spg(batch, self.actor, \
                                self.critic, sigma, explore, self.device, self.version)

                    else:
                        train_states_v, train_actions_v, train_rewards_v, \
                        train_dones_mask, train_last_states_v = \
                            common_torch.unpack_batch(batch, self.device)

                    if self.version == 'DDPG':
                        # train critic
                        self.critic_optim.zero_grad()
                        last_act_v = self.actor_tgt.target_model(
                            train_last_states_v)
                        q_v = self.critic(train_states_v, train_actions_v)
                        q_last_v = self.critic_tg.target_model(
                            train_last_states_v, last_act_v)
                        q_last_v[train_dones_mask] = 0.0
                        q_ref_v = train_rewards_v.unsqueeze(dim=-1) + \
                                q_last_v * self.gamma ** self.roll_steps
                        l = (q_v - q_ref_v) ** 2 
                        if self.prio:
                            critic_loss_v = batch_weights_v * l
                            prios = (critic_loss_v.detach() + 1e-5)
                        else:
                            critic_loss_v = l
                        critic_loss_v = torch.mean(critic_loss_v)
                        critic_loss_v.backward()
                        self.critic_optim.step()
                        tb_tracker.track("loss_critic",
                                        critic_loss_v, self.frame_idx)
                        tb_tracker.track("Q_target",
                                        q_ref_v.mean(), self.frame_idx)
                        tb_tracker.track("Q", q_v, self.frame_idx)
                        # train actor
                        self.actor_optim.zero_grad()
                        cur_actions_v = self.actor(train_states_v)
                        actor_loss_v = - self.critic(train_states_v, cur_actions_v)
                        actor_loss_v = actor_loss_v.mean()
                        actor_loss_v.backward()
                        self.actor_optim.step()
                        tb_tracker.track("loss_actor",
                                        actor_loss_v, self.frame_idx)
                        #Perform soft-weight sync
                        self.actor_tgt.alpha_sync(alpha=1 - 1e-3)
                        self.critic_tg.alpha_sync(alpha=1 - 1e-3)

                    elif self.version == 'SPG':
                        # train critic
                        self.critic_optim.zero_grad()
                        q_v = self.critic(train_states_v, train_actions_v)
                        q_last_v = self.critic_tg.target_model(
                            train_last_states_v, best_actions)
                        q_last_v[train_dones_mask] = 0.0
                        q_ref_v = train_rewards_v.unsqueeze(dim=-1) + \
                                q_last_v * self.gamma ** self.roll_steps
                        l = (q_v - q_ref_v) ** 2 
                        if self.prio:
                            critic_loss_v = batch_weights_v * l
                            prios = (critic_loss_v.detach() + 1e-5)
                        else:
                            critic_loss_v = l
                        critic_loss_v = torch.mean(critic_loss_v)
                        critic_loss_v.backward()
                        self.critic_optim.step()
                        tb_tracker.track("loss_critic",
                                        critic_loss_v, self.frame_idx)
                        tb_tracker.track("Q_target",
                                        q_ref_v.mean(), self.frame_idx)
                        tb_tracker.track("Q", q_v, self.frame_idx)
                        # train actor
                        self.actor_optim.zero_grad()
                        cur_actions_v = self.actor(train_states_v)
                        # actor_loss_v = - self.critic(train_states_v, cur_actions_v)
                        # actor_loss_v = actor_loss_v.mean()
                        actor_loss_v = F.mse_loss(cur_actions_v, best_actions)
                        actor_loss_v.backward()
                        self.actor_optim.step()
                        tb_tracker.track("loss_actor",
                                        actor_loss_v, self.frame_idx)
                        #Perform soft-weight sync
                        self.actor_tgt.alpha_sync(alpha=1 - 1e-3)
                        self.critic_tg.alpha_sync(alpha=1 - 1e-3)

                    elif self.version == 'TD3':
                        with torch.no_grad():
                            #Add clipped noise
                            noise = (
                                torch.randn_like(train_actions_v) * self.policy_noise
                                ).clamp(-self.noise_clip, self.noise_clip)
                            next_action = (
                                self.actor_tgt(train_last_states_v) + noise
                                ).clamp(-1, 1)
                            # Compute the target Q value
                            target_Q1, target_Q2 = self.critic_tg(train_last_states_v, next_action)
                            target_Q = torch.min(target_Q1, target_Q2)
                            target_Q[train_dones_mask] = 0.0
                            target_Q = train_rewards_v.unsqueeze(dim=-1) + target_Q*self.gamma**self.roll_steps 
                            tb_tracker.track("Q_target", target_Q, self.frame_idx)
                    
                        current_Q1, current_Q2 = self.critic(train_states_v, train_actions_v)
                        # Optimize the critic
                        self.critic_optim.zero_grad()
                        l1 = (current_Q1 - target_Q)**2
                        l2 = (current_Q2 - target_Q)**2
                        if self.prio:
                            critic_loss = batch_weights_v * (l1 + l2)
                            prios = (critic_loss.detach() + 1e-5)
                        else:
                            critic_loss = l1 + l2
                        critic_loss = torch.mean(critic_loss)
                        critic_loss.backward()
                        self.critic_optim.step()
                        tb_tracker.track("loss_critic", critic_loss, self.frame_idx)
                        tb_tracker.track("Q1", current_Q1.mean(), self.frame_idx)
                        tb_tracker.track("Q2", current_Q2.mean(), self.frame_idx)
                        #Delayed policy updates
                        if self.frame_idx % self.policy_freq == 0:
                            self.actor_optim.zero_grad()
                            mu = self.actor(train_states_v)
                            Q, _ = self.critic(train_states_v, mu)
                            actor_loss_v = - Q.mean()
                            actor_loss_v.backward()
                            self.actor_optim.step()
                            tb_tracker.track("loss_actor", actor_loss_v, self.frame_idx)

                            for param, target_param in zip(self.critic.parameters(), self.critic_tg.parameters()):
                                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                            for param, target_param in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    elif self.version == 'SPGR':
                        with torch.no_grad():
                            #Add clipped noise
                            noise = (
                                torch.randn_like(train_actions_v) * self.policy_noise
                                ).clamp(-self.noise_clip, self.noise_clip)
                            next_action = (
                                self.actor_tgt(train_last_states_v) + noise
                                ).clamp(-1, 1)
                            # Compute the target Q value
                            target_Q1, target_Q2 = self.critic_tg(train_last_states_v, next_action)
                            target_Q = torch.min(target_Q1, target_Q2)
                            target_Q[train_dones_mask] = 0.0
                            target_Q = train_rewards_v.unsqueeze(dim=-1) + target_Q*self.gamma**self.roll_steps 
                            tb_tracker.track("Q_target", target_Q, self.frame_idx)
                    
                        current_Q1, current_Q2 = self.critic(train_states_v, train_actions_v)
                        # Optimize the critic
                        self.critic_optim.zero_grad()
                        l1 = (current_Q1 - target_Q)**2
                        l2 = (current_Q2 - target_Q)**2
                        if self.prio:
                            critic_loss = batch_weights_v * (l1 + l2)
                            prios = (critic_loss.detach() + 1e-5)
                        else:
                            critic_loss = l1 + l2
                        critic_loss = torch.mean(critic_loss)
                        critic_loss.backward()
                        self.critic_optim.step()
                        tb_tracker.track("loss_critic", critic_loss, self.frame_idx)
                        tb_tracker.track("Q1", current_Q1.mean(), self.frame_idx)
                        tb_tracker.track("Q2", current_Q2.mean(), self.frame_idx)
                        #Delayed policy updates
                        if self.frame_idx % self.policy_freq == 0:
                            self.actor_optim.zero_grad()
                            mu = self.actor(train_states_v)
                            Q, _ = self.critic(train_states_v, mu)
                            lamda = self.alpha = self.alpha/(Q.abs().mean().detach())
                            actor_loss_v = -lamda * Q.mean() + F.mse_loss(mu, best_actions)
                            actor_loss_v.backward()
                            self.actor_optim.step()
                            tb_tracker.track("loss_actor", actor_loss_v, self.frame_idx)

                            for param, target_param in zip(self.critic.parameters(), self.critic_tg.parameters()):
                                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                            for param, target_param in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    
                    if self.prio:
                        self.buffer.update_priorities(batch_indices, prios.cpu().numpy())
                    else:
                        pass             
                    #Test handling point
                    if self.frame_idx % self.test_iters == 0:
                        ts = time.time()
                        rewards, steps = test_net(self.actor, self.test_env, device=self.device)
                        print("Test done in %.2f sec, reward %.3f, steps %d" % (
                            time.time() - ts, rewards, steps))
                        self.writer.add_scalar("test_reward", rewards, self.frame_idx)
                        self.writer.add_scalar("test_steps", steps, self.frame_idx)
                        if self.best_reward is None or self.best_reward < rewards:
                            if self.best_reward is not None:
                                print("Best reward updated: %.3f -> %.3f" % (self.best_reward, rewards))
                                name = "best_%+.3f_%d.dat" % (rewards, self.frame_idx)
                                act_name = os.path.join(self.save_path, name)
                                crt_name = os.path.join(self.save_path, "Q_" + name)
                                torch.save(self.actor.state_dict(), act_name)
                                torch.save(self.critic.state_dict(), crt_name)
                            self.best_reward = rewards
                    elif self.frame_idx == stop:
                        return