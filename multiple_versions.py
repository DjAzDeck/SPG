import copy
import os
import random
import time
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import ptan
from lib import common_torch, spg_torch

from tqdm.auto import tqdm

def test_net(net: nn.Module, env: gym.Env, seed: int, count: int = 10, device: str = "cpu"):
    rewards = 0.0
    steps = 0
    for i in range(count):
        obs, _ = env.reset(seed=seed + i)
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_s = net(obs_v)
            action = mu_s.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, truncated, _ = env.step(action)
            rewards += reward
            steps += 1
            if done or truncated:
                break
    return rewards / count, steps / count


class Trainer:
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
        seed: int = 2025,
        replay_size: int = 1000000,
        discount: float = 0.99,
        LR: float = 1e-4,
        prio_alpha: float = 0.6,
        rollout_step: int = 1,
        policy_freq: int = 2,
        noise_clip: float = 0.5,
        policy_noise: float = 0.2,
        alpha: float = 2.5,
        tau: float = 0.005,
        sigma_start: float = 0.7,
        sigma_final: float = 0.01,
        sigma_decay_last_frame: int = 1200000,
        beta_start: float = 0.4,
        beta_frames: int = 400000,
        replay_initial: int = 5000,
        test_iters: int = 5000,
        num_envs: int = 1,
        checkpoint_interval: int = 50000,
        checkpoint_load_path: str | None = None,
        resume_from_checkpoint: bool = False,
        save_replay_buffer: bool = True,
        replay_save_interval: int = 50000,
        replay_load_path: str | None = None,
        search_final: int | None = None,
        search_decay_last_frame: int = 0,
        search_chunk_size: int = 0,
        summary_interval_sec: float = 10.0,
        tqdm_enabled: bool = True,
        eval_episodes: int = 10,
    ):
        self.LR = LR
        self.device = device
        self.prio = bool(prio)
        self.seed = seed
        self.prio_alpha = prio_alpha
        self.batch_size = batch_size
        self.gamma = discount
        self.roll_steps = rollout_step
        self.env = env
        self.test_env = test_env
        self.replay_initial = replay_initial
        self.test_iters = test_iters
        self.version = version
        self.policy_freq = policy_freq
        self.policy_noise = policy_noise
        self.alpha = alpha
        self.tau = tau
        self.noise_clip = noise_clip
        self.sigma_start = sigma_start
        self.sigma_final = sigma_final
        self.sigma_decay_last_frame = sigma_decay_last_frame
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.num_envs = max(1, int(num_envs))
        self.checkpoint_interval = max(0, int(checkpoint_interval))
        self.checkpoint_load_path = checkpoint_load_path
        self.resume_from_checkpoint = bool(resume_from_checkpoint)
        self.save_replay_buffer_enabled = bool(save_replay_buffer)
        self.replay_save_interval = max(0, int(replay_save_interval))
        self.replay_load_path = replay_load_path
        self.search_final = None if search_final is None else max(0, int(search_final))
        self.search_decay_last_frame = max(0, int(search_decay_last_frame))
        self.search_chunk_size = max(0, int(search_chunk_size))
        self.summary_interval_sec = max(1.0, float(summary_interval_sec))
        self.tqdm_enabled = bool(tqdm_enabled)
        self.eval_episodes = max(1, int(eval_episodes))
        self.run_dir: str | None = None
        self.save_path: str | None = None
        self.replay_path: str | None = None
        self.frame_idx = 0
        self.best_reward = None

        # Init nets
        if self.version == "DDPG":
            self.actor = spg_torch.SPGActor(state_dim, action_dim).to(self.device)
            self.actor_tgt = ptan.agent.TargetNet(self.actor)
            self.critic = spg_torch.SPGCritic(state_dim, action_dim).to(self.device)
            self.critic_tg = ptan.agent.TargetNet(self.critic)
            self.agent = spg_torch.AgentDDPG(self.actor, device=self.device)
        elif self.version == "SPG":
            self.actor = spg_torch.SPGActor(state_dim, action_dim).to(device)
            self.agent = spg_torch.AgentSPG(self.actor, device=self.device)
            self.actor_tgt = ptan.agent.TargetNet(self.actor)
            self.critic = spg_torch.SPGCritic(state_dim, action_dim).to(device)
            self.critic_tg = ptan.agent.TargetNet(self.critic)
        elif (self.version == "TD3") or (self.version == "SPGR"):
            self.actor = spg_torch.SPGActor(state_dim, action_dim).to(device)
            self.agent = spg_torch.AgentSPG(self.actor, device=self.device)
            self.actor_tgt = copy.deepcopy(self.actor)
            self.critic = spg_torch.Critic(state_dim, action_dim).to(device)
            self.critic_tg = copy.deepcopy(self.critic)
        else:
            raise ValueError(f"Unsupported version: {self.version}")
        print(self.actor)
        print(self.critic)

        # Init optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.LR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.LR)

        # Init experience source & replay buffer
        if isinstance(env, gym.vector.VectorEnv):
            self.experience_source = ptan.experience.VectorExperienceSourceFirstLast(
                env, self.agent, gamma=discount, env_seed=self.seed, steps_count=1, unnest_data=True
            )
            self.populate_per_iter = self.num_envs
        else:
            self.experience_source = ptan.experience.ExperienceSourceFirstLast(
                env, self.agent, gamma=discount, env_seed=self.seed, steps_count=1
            )
            self.populate_per_iter = 1

        if self.prio:
            self.buffer = ptan.experience.PrioritizedReplayBuffer(
                self.experience_source, buffer_size=replay_size, alpha=self.prio_alpha
            )
        else:
            self.buffer = ptan.experience.ExperienceReplayBuffer(
                self.experience_source, buffer_size=replay_size
            )

        if self.replay_load_path:
            self.load_replay_buffer(self.replay_load_path)

        if self.checkpoint_load_path:
            self.load_checkpoint(self.checkpoint_load_path, resume=self.resume_from_checkpoint)

    def _beta_by_frame(self, frame_idx: int) -> float:
        denom = max(self.beta_frames, 1)
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / denom)

    def _sigma_by_frame(self, frame_idx: int) -> float:
        decay = (self.sigma_start - self.sigma_final) * frame_idx / max(self.sigma_decay_last_frame, 1)
        return max(self.sigma_final, self.sigma_start - decay)

    def _search_budget(self, start_search: int, frame_idx: int) -> int:
        start = max(0, int(start_search))
        if self.search_final is None:
            return start
        final = self.search_final
        if self.search_decay_last_frame <= 0:
            return final
        ratio = min(1.0, frame_idx / self.search_decay_last_frame)
        return max(0, int(round(start + (final - start) * ratio)))

    @staticmethod
    def _console_log(message: str, progress_bar=None):
        if progress_bar is not None:
            progress_bar.write(message)
        else:
            print(message)

    @staticmethod
    def _target_model(target: Any) -> nn.Module:
        if hasattr(target, "target_model"):
            return target.target_model
        return target

    def _capture_rng_state(self) -> dict:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: dict):
        if not state:
            return
        if state.get("python") is not None:
            random.setstate(state["python"])
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
        if state.get("torch") is not None:
            torch.set_rng_state(state["torch"])
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _checkpoint_payload(self) -> dict:
        return {
            "checkpoint_version": 1,
            "algorithm_version": self.version,
            "frame_idx": self.frame_idx,
            "best_reward": self.best_reward,
            "num_envs": self.num_envs,
            "prioritized_replay": self.prio,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self._target_model(self.actor_tgt).state_dict(),
            "critic_target": self._target_model(self.critic_tg).state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "rng_state": self._capture_rng_state(),
        }

    def save_checkpoint(self, tag: str | None = None) -> str:
        if not self.save_path:
            raise RuntimeError("save_path is not initialized. Call train_routine() first.")
        checkpoint_name = f"checkpoint_{self.frame_idx:09d}.pt" if tag is None else f"{tag}.pt"
        checkpoint_path = os.path.join(self.save_path, checkpoint_name)
        torch.save(self._checkpoint_payload(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str, resume: bool):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "actor" not in checkpoint:
            # Backward compatibility: legacy checkpoints only stored actor weights.
            if resume:
                raise ValueError("Cannot resume from legacy actor-only checkpoint")
            self.actor.load_state_dict(checkpoint)
            self._target_model(self.actor_tgt).load_state_dict(self.actor.state_dict())
            legacy_critic_path = os.path.join(
                os.path.dirname(checkpoint_path), f"Q_{os.path.basename(checkpoint_path)}"
            )
            if os.path.exists(legacy_critic_path):
                critic_state = torch.load(legacy_critic_path, map_location=self.device, weights_only=False)
                self.critic.load_state_dict(critic_state)
                self._target_model(self.critic_tg).load_state_dict(self.critic.state_dict())
                print(f"Loaded legacy actor+critic checkpoint from {checkpoint_path}")
            else:
                print(f"Loaded legacy actor-only checkpoint from {checkpoint_path}")
            return

        ckpt_version = checkpoint.get("algorithm_version")
        if ckpt_version and ckpt_version != self.version:
            print(f"Warning: loading checkpoint version={ckpt_version} into current version={self.version}")

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self._target_model(self.actor_tgt).load_state_dict(checkpoint["actor_target"])
        self._target_model(self.critic_tg).load_state_dict(checkpoint["critic_target"])

        if resume:
            self.actor_optim.load_state_dict(checkpoint["actor_optim"])
            self.critic_optim.load_state_dict(checkpoint["critic_optim"])
            self.frame_idx = int(checkpoint.get("frame_idx", 0))
            self.best_reward = checkpoint.get("best_reward")
            self._restore_rng_state(checkpoint.get("rng_state", {}))
            print(f"Resumed checkpoint from {checkpoint_path} at frame {self.frame_idx}")
        else:
            print(f"Loaded model weights from checkpoint {checkpoint_path}")

    def save_replay_buffer(self, tag: str | None = None) -> str:
        if not self.replay_path:
            raise RuntimeError("replay_path is not initialized. Call train_routine() first.")
        replay_name = f"replay_{self.frame_idx:09d}.pt" if tag is None else f"replay_{tag}.pt"
        replay_path = os.path.join(self.replay_path, replay_name)
        payload = {
            "snapshot_version": 1,
            "algorithm_version": self.version,
            "frame_idx": self.frame_idx,
            "prioritized_replay": self.prio,
            "buffer_state": self.buffer.state_dict(),
        }
        torch.save(payload, replay_path)
        return replay_path

    def load_replay_buffer(self, replay_path: str):
        payload = torch.load(replay_path, map_location="cpu", weights_only=False)
        buffer_state = payload.get("buffer_state", payload)
        state_kind = buffer_state.get("type", "uniform")
        expected_kind = "prioritized" if self.prio else "uniform"
        if state_kind != expected_kind:
            raise ValueError(
                f"Replay buffer type mismatch: checkpoint={state_kind} current={expected_kind}"
            )
        self.buffer.load_state_dict(buffer_state)
        print(f"Loaded replay buffer from {replay_path} with {len(self.buffer)} items")

    @staticmethod
    def _next_trigger_frame(current: int, interval: int) -> int:
        if interval <= 0:
            return 0
        return ((current // interval) + 1) * interval

    def train_routine(
        self,
        name: str,
        explore: int,
        stop: int,
        run_dir: str | None = None,
        tags: list[str] | None = None,
    ):
        self.run_dir = run_dir or os.path.join("runs", name)
        self.save_path = os.path.join(self.run_dir, "checkpoints")
        self.replay_path = os.path.join(self.run_dir, "replay")
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.replay_path, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "logs"), comment="-" + name)
        if tags:
            self.writer.add_text("run/tags", ",".join(tags), global_step=self.frame_idx)
        if self.checkpoint_load_path:
            self.writer.add_text("run/checkpoint_load_path", self.checkpoint_load_path, global_step=self.frame_idx)
            self.writer.add_scalar("run/checkpoint_resume", int(self.resume_from_checkpoint), self.frame_idx)
        if self.replay_load_path:
            self.writer.add_text("run/replay_load_path", self.replay_load_path, global_step=self.frame_idx)
            self.writer.add_scalar("run/replay_size_loaded", len(self.buffer), self.frame_idx)

        next_test_frame = self._next_trigger_frame(self.frame_idx, self.test_iters)
        next_checkpoint_frame = self._next_trigger_frame(self.frame_idx, self.checkpoint_interval)
        next_replay_frame = self._next_trigger_frame(self.frame_idx, self.replay_save_interval)
        progress_bar = None
        if self.tqdm_enabled and tqdm is not None:
            progress_bar = tqdm(
                total=stop,
                initial=min(self.frame_idx, stop),
                dynamic_ncols=True,
                unit="step",
                desc="train",
            )

        train_start = time.perf_counter()
        eval_time_total = 0.0
        last_summary_time = 0.0
        last_summary_frame = self.frame_idx
        last_summary_updates = 0
        last_summary_actor_updates = 0
        last_summary_episodes = 0
        updates_total = 0
        actor_updates_total = 0
        episodes_total = 0
        recent_rewards = deque(maxlen=100)
        last_actor_loss = float("nan")
        last_critic_loss = float("nan")
        last_search_budget = self._search_budget(explore, self.frame_idx)

        try:
            with ptan.common.utils.RewardTracker(self.writer, min_ts_diff=float("inf")) as tracker:
                with ptan.common.utils.TBMeanTracker(self.writer, batch_size=10) as tb_tracker:
                    while self.frame_idx < stop:
                        self.buffer.populate(self.populate_per_iter)
                        self.frame_idx += self.populate_per_iter
                        if progress_bar is not None:
                            progress_bar.update(max(0, min(self.frame_idx, stop) - progress_bar.n))

                        rewards_steps = self.experience_source.pop_rewards_steps()
                        if rewards_steps:
                            episodes_total += len(rewards_steps)
                            for reward, steps in rewards_steps:
                                recent_rewards.append(float(reward))
                                tb_tracker.track("episode_steps", steps, self.frame_idx)
                                tracker.reward(reward, self.frame_idx)

                        if self.checkpoint_interval > 0 and self.frame_idx >= next_checkpoint_frame:
                            self.save_checkpoint(tag=f"step_{next_checkpoint_frame:09d}")
                            while self.frame_idx >= next_checkpoint_frame:
                                next_checkpoint_frame += self.checkpoint_interval

                        if (
                            self.save_replay_buffer_enabled
                            and self.replay_save_interval > 0
                            and self.frame_idx >= next_replay_frame
                        ):
                            self.save_replay_buffer(tag=f"step_{next_replay_frame:09d}")
                            while self.frame_idx >= next_replay_frame:
                                next_replay_frame += self.replay_save_interval

                        if len(self.buffer) < self.replay_initial:
                            continue

                        sigma = round(self._sigma_by_frame(self.frame_idx), 3)
                        last_search_budget = self._search_budget(explore, self.frame_idx)
                        beta = self._beta_by_frame(self.frame_idx) if self.prio else 1.0

                        if self.prio:
                            batch, batch_indices, batch_weights = self.buffer.sample(self.batch_size, beta)
                            batch_weights_v = torch.tensor(batch_weights, device=self.device).unsqueeze(1)
                        else:
                            batch = self.buffer.sample(self.batch_size)

                        best_actions = None
                        use_search = False
                        if (self.version == "SPG") or (self.version == "SPGR"):
                            use_search = (self.version == "SPG") or (self.frame_idx % self.policy_freq == 0)
                            if use_search:
                                (
                                    train_states_v,
                                    train_actions_v,
                                    train_rewards_v,
                                    train_dones_mask,
                                    train_last_states_v,
                                    best_actions,
                                ) = common_torch.unpack_batch_spg(
                                    batch,
                                    self.actor,
                                    self.critic,
                                    sigma,
                                    last_search_budget,
                                    self.device,
                                    self.version,
                                    search_chunk_size=self.search_chunk_size,
                                )
                            else:
                                (
                                    train_states_v,
                                    train_actions_v,
                                    train_rewards_v,
                                    train_dones_mask,
                                    train_last_states_v,
                                ) = common_torch.unpack_batch(batch, self.device)
                        else:
                            (
                                train_states_v,
                                train_actions_v,
                                train_rewards_v,
                                train_dones_mask,
                                train_last_states_v,
                            ) = common_torch.unpack_batch(batch, self.device)

                        updates_total += 1
                        if self.version == "DDPG":
                            self.critic_optim.zero_grad()
                            last_act_v = self.actor_tgt.target_model(train_last_states_v)
                            q_v = self.critic(train_states_v, train_actions_v)
                            q_last_v = self.critic_tg.target_model(train_last_states_v, last_act_v)
                            q_last_v[train_dones_mask] = 0.0
                            q_ref_v = train_rewards_v.unsqueeze(dim=-1) + q_last_v * self.gamma ** self.roll_steps
                            l = (q_v - q_ref_v) ** 2
                            if self.prio:
                                critic_loss_v = batch_weights_v * l
                                prios = critic_loss_v.detach() + 1e-5
                            else:
                                critic_loss_v = l
                            critic_loss_v = torch.mean(critic_loss_v)
                            last_critic_loss = float(critic_loss_v.item())
                            critic_loss_v.backward()
                            self.critic_optim.step()
                            tb_tracker.track("loss_critic", critic_loss_v, self.frame_idx)
                            tb_tracker.track("Q_target", q_ref_v.mean(), self.frame_idx)
                            tb_tracker.track("Q", q_v, self.frame_idx)

                            self.actor_optim.zero_grad()
                            cur_actions_v = self.actor(train_states_v)
                            actor_loss_v = -self.critic(train_states_v, cur_actions_v)
                            actor_loss_v = actor_loss_v.mean()
                            last_actor_loss = float(actor_loss_v.item())
                            actor_loss_v.backward()
                            self.actor_optim.step()
                            actor_updates_total += 1
                            tb_tracker.track("loss_actor", actor_loss_v, self.frame_idx)

                            self.actor_tgt.alpha_sync(alpha=1 - 1e-3)
                            self.critic_tg.alpha_sync(alpha=1 - 1e-3)

                        elif self.version == "SPG":
                            self.critic_optim.zero_grad()
                            q_v = self.critic(train_states_v, train_actions_v)
                            q_last_v = self.critic_tg.target_model(train_last_states_v, best_actions)
                            q_last_v[train_dones_mask] = 0.0
                            q_ref_v = train_rewards_v.unsqueeze(dim=-1) + q_last_v * self.gamma ** self.roll_steps
                            l = (q_v - q_ref_v) ** 2
                            if self.prio:
                                critic_loss_v = batch_weights_v * l
                                prios = critic_loss_v.detach() + 1e-5
                            else:
                                critic_loss_v = l
                            critic_loss_v = torch.mean(critic_loss_v)
                            last_critic_loss = float(critic_loss_v.item())
                            critic_loss_v.backward()
                            self.critic_optim.step()
                            tb_tracker.track("loss_critic", critic_loss_v, self.frame_idx)
                            tb_tracker.track("Q_target", q_ref_v.mean(), self.frame_idx)
                            tb_tracker.track("Q", q_v, self.frame_idx)

                            self.actor_optim.zero_grad()
                            cur_actions_v = self.actor(train_states_v)
                            actor_loss_v = F.mse_loss(cur_actions_v, best_actions)
                            last_actor_loss = float(actor_loss_v.item())
                            actor_loss_v.backward()
                            self.actor_optim.step()
                            actor_updates_total += 1
                            tb_tracker.track("loss_actor", actor_loss_v, self.frame_idx)

                            self.actor_tgt.alpha_sync(alpha=1 - 1e-3)
                            self.critic_tg.alpha_sync(alpha=1 - 1e-3)

                        elif self.version == "TD3":
                            with torch.no_grad():
                                noise = (torch.randn_like(train_actions_v) * self.policy_noise).clamp(
                                    -self.noise_clip, self.noise_clip
                                )
                                next_action = (self.actor_tgt(train_last_states_v) + noise).clamp(-1, 1)
                                target_Q1, target_Q2 = self.critic_tg(train_last_states_v, next_action)
                                target_Q = torch.min(target_Q1, target_Q2)
                                target_Q[train_dones_mask] = 0.0
                                target_Q = train_rewards_v.unsqueeze(dim=-1) + target_Q * self.gamma ** self.roll_steps
                                tb_tracker.track("Q_target", target_Q, self.frame_idx)

                            current_Q1, current_Q2 = self.critic(train_states_v, train_actions_v)
                            self.critic_optim.zero_grad()
                            l1 = (current_Q1 - target_Q) ** 2
                            l2 = (current_Q2 - target_Q) ** 2
                            if self.prio:
                                critic_loss = batch_weights_v * (l1 + l2)
                                prios = critic_loss.detach() + 1e-5
                            else:
                                critic_loss = l1 + l2
                            critic_loss = torch.mean(critic_loss)
                            last_critic_loss = float(critic_loss.item())
                            critic_loss.backward()
                            self.critic_optim.step()
                            tb_tracker.track("loss_critic", critic_loss, self.frame_idx)
                            tb_tracker.track("Q1", current_Q1.mean(), self.frame_idx)
                            tb_tracker.track("Q2", current_Q2.mean(), self.frame_idx)

                            if self.frame_idx % self.policy_freq == 0:
                                self.actor_optim.zero_grad()
                                mu = self.actor(train_states_v)
                                Q, _ = self.critic(train_states_v, mu)
                                actor_loss_v = -Q.mean()
                                last_actor_loss = float(actor_loss_v.item())
                                actor_loss_v.backward()
                                self.actor_optim.step()
                                actor_updates_total += 1
                                tb_tracker.track("loss_actor", actor_loss_v, self.frame_idx)

                                for param, target_param in zip(self.critic.parameters(), self.critic_tg.parameters()):
                                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                                for param, target_param in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                        elif self.version == "SPGR":
                            with torch.no_grad():
                                noise = (torch.randn_like(train_actions_v) * self.policy_noise).clamp(
                                    -self.noise_clip, self.noise_clip
                                )
                                next_action = (self.actor_tgt(train_last_states_v) + noise).clamp(-1, 1)
                                target_Q1, target_Q2 = self.critic_tg(train_last_states_v, next_action)
                                target_Q = torch.min(target_Q1, target_Q2)
                                target_Q[train_dones_mask] = 0.0
                                target_Q = train_rewards_v.unsqueeze(dim=-1) + target_Q * self.gamma ** self.roll_steps
                                tb_tracker.track("Q_target", target_Q, self.frame_idx)

                            current_Q1, current_Q2 = self.critic(train_states_v, train_actions_v)
                            self.critic_optim.zero_grad()
                            l1 = (current_Q1 - target_Q) ** 2
                            l2 = (current_Q2 - target_Q) ** 2
                            if self.prio:
                                critic_loss = batch_weights_v * (l1 + l2)
                                prios = critic_loss.detach() + 1e-5
                            else:
                                critic_loss = l1 + l2
                            critic_loss = torch.mean(critic_loss)
                            last_critic_loss = float(critic_loss.item())
                            critic_loss.backward()
                            self.critic_optim.step()
                            tb_tracker.track("loss_critic", critic_loss, self.frame_idx)
                            tb_tracker.track("Q1", current_Q1.mean(), self.frame_idx)
                            tb_tracker.track("Q2", current_Q2.mean(), self.frame_idx)

                            if self.frame_idx % self.policy_freq == 0:
                                if best_actions is None:
                                    raise RuntimeError("SPGR actor update expected best_actions but got None")
                                self.actor_optim.zero_grad()
                                mu = self.actor(train_states_v)
                                Q, _ = self.critic(train_states_v, mu)
                                q_scale = Q.abs().mean().detach().clamp_min(1e-6)
                                lamda = torch.clamp(self.alpha / q_scale, 0.1, 10.0)
                                actor_loss_v = -lamda * Q.mean() + F.mse_loss(mu, best_actions)
                                last_actor_loss = float(actor_loss_v.item())
                                actor_loss_v.backward()
                                self.actor_optim.step()
                                actor_updates_total += 1
                                tb_tracker.track("loss_actor", actor_loss_v, self.frame_idx)

                                for param, target_param in zip(self.critic.parameters(), self.critic_tg.parameters()):
                                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                                for param, target_param in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                        if self.prio:
                            prios_np = prios.detach().view(-1).clamp_min(1e-6).cpu().numpy()
                            self.buffer.update_priorities(batch_indices, prios_np)

                        if self.test_iters > 0 and self.frame_idx >= next_test_frame:
                            eval_start = time.perf_counter()
                            rewards, steps = test_net(
                                self.actor,
                                self.test_env,
                                seed=self.seed + 256,
                                count=self.eval_episodes,
                                device=self.device,
                            )
                            eval_elapsed = time.perf_counter() - eval_start
                            eval_time_total += eval_elapsed
                            self.writer.add_scalar("test_reward", rewards, self.frame_idx)
                            self.writer.add_scalar("test_steps", steps, self.frame_idx)
                            self.writer.add_scalar("test_time_sec", eval_elapsed, self.frame_idx)
                            best_status = "unchanged"
                            if self.best_reward is None or self.best_reward < rewards:
                                prev_best = self.best_reward
                                self.best_reward = rewards
                                legacy_name = "best_%+.3f_%d.dat" % (rewards, self.frame_idx)
                                act_name = os.path.join(self.save_path, legacy_name)
                                crt_name = os.path.join(self.save_path, "Q_" + legacy_name)
                                torch.save(self.actor.state_dict(), act_name)
                                torch.save(self.critic.state_dict(), crt_name)
                                self.save_checkpoint(tag=f"best_{self.frame_idx:09d}")
                                best_status = (
                                    f"initialized={rewards:.3f}"
                                    if prev_best is None
                                    else f"updated={prev_best:.3f}->{rewards:.3f}"
                                )
                            self._console_log(
                                (
                                    f"[eval] step={self.frame_idx} reward={rewards:.3f} "
                                    f"steps={int(steps)} dt={eval_elapsed:.2f}s best={best_status}"
                                ),
                                progress_bar,
                            )
                            while self.frame_idx >= next_test_frame:
                                next_test_frame += self.test_iters

                        now = time.perf_counter()
                        effective_time = max(1e-6, now - train_start - eval_time_total)
                        if effective_time - last_summary_time >= self.summary_interval_sec:
                            delta_t = max(1e-6, effective_time - last_summary_time)
                            frame_delta = self.frame_idx - last_summary_frame
                            update_delta = updates_total - last_summary_updates
                            actor_update_delta = actor_updates_total - last_summary_actor_updates
                            episode_delta = episodes_total - last_summary_episodes
                            env_steps_per_sec = frame_delta / delta_t
                            updates_per_sec = update_delta / delta_t
                            actor_updates_per_sec = actor_update_delta / delta_t
                            episodes_per_sec = episode_delta / delta_t
                            reward_100 = float(np.mean(recent_rewards)) if recent_rewards else float("nan")
                            replay_fill_pct = 100.0 * len(self.buffer) / max(1, self.buffer.capacity)
                            summary = (
                                f"[train] step={self.frame_idx}/{stop} "
                                f"env_sps={env_steps_per_sec:.1f} upd_sps={updates_per_sec:.1f} "
                                f"act_upd_sps={actor_updates_per_sec:.2f} eps={episodes_total} "
                                f"eps_s={episodes_per_sec:.2f} replay={len(self.buffer)}/{self.buffer.capacity} "
                                f"({replay_fill_pct:.1f}%) sigma={sigma:.3f} beta={beta:.3f} "
                                f"search={last_search_budget if use_search else 0} "
                                f"loss_c={last_critic_loss:.4f} loss_a={last_actor_loss:.4f} "
                                f"reward100={reward_100:.3f}"
                            )
                            self._console_log(summary, progress_bar)
                            self.writer.add_scalar("cli/env_steps_per_sec", env_steps_per_sec, self.frame_idx)
                            self.writer.add_scalar("cli/updates_per_sec", updates_per_sec, self.frame_idx)
                            self.writer.add_scalar("cli/actor_updates_per_sec", actor_updates_per_sec, self.frame_idx)
                            self.writer.add_scalar("cli/episodes_per_sec", episodes_per_sec, self.frame_idx)
                            self.writer.add_scalar("cli/replay_fill_percent", replay_fill_pct, self.frame_idx)
                            self.writer.add_scalar("cli/search_budget", last_search_budget if use_search else 0, self.frame_idx)
                            self.writer.add_scalar("cli/sigma", sigma, self.frame_idx)
                            self.writer.add_scalar("cli/beta", beta, self.frame_idx)

                            if progress_bar is not None:
                                progress_bar.set_postfix(
                                    {
                                        "env_sps": f"{env_steps_per_sec:.0f}",
                                        "upd_sps": f"{updates_per_sec:.0f}",
                                        "search": int(last_search_budget if use_search else 0),
                                        "replay%": f"{replay_fill_pct:.0f}",
                                    },
                                    refresh=False,
                                )
                            last_summary_time = effective_time
                            last_summary_frame = self.frame_idx
                            last_summary_updates = updates_total
                            last_summary_actor_updates = actor_updates_total
                            last_summary_episodes = episodes_total
        finally:
            if progress_bar is not None:
                progress_bar.close()

        self.save_checkpoint(tag="last")
        if self.save_replay_buffer_enabled:
            self.save_replay_buffer(tag="last")
