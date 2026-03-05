import gymnasium as gym
import torch
import random
import collections
import typing as tt
from dataclasses import dataclass

import numpy as np

from collections import deque

from .agent import BaseAgent
from .common import utils

State = np.ndarray
Action = int


@dataclass(frozen=True)
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """
    Item = tt.Tuple[Experience, ...]

    def __init__(self, env: gym.Env | tt.Collection[gym.Env], agent: BaseAgent,
                 steps_count: int = 2, steps_delta: int = 1,
                 env_seed: tt.Optional[int] = None):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param env_seed: seed to be used in Env.reset() call
        """
        assert steps_count >= 1
        if isinstance(env, (list, tuple)):
            self.pool = env
            # do the check for the multiple copies passed
            ids = set(id(e) for e in env)
            if len(ids) < len(env):
                raise ValueError("You passed single environment instance multiple times")
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.agent_states = [agent.initial_state() for _ in self.pool]
        self.env_seed = env_seed
        self._next_seed = env_seed

    def _reset_env(self, env: gym.Env) -> tuple[tt.Any, dict]:
        if self._next_seed is None:
            return env.reset()
        seed = int(self._next_seed)
        self._next_seed += 1
        return env.reset(seed=seed)

    def __iter__(self) -> tt.Generator[Item, None, None]:
        states, histories, cur_rewards, cur_steps = [], [], [], []
        for env in self.pool:
            obs, _ = self._reset_env(env)
            states.append(obs)
            histories.append(deque(maxlen=self.steps_count))
            cur_rewards.append(0.0)
            cur_steps.append(0)

        iter_idx = 0
        while True:
            actions, self.agent_states = self.agent(states, self.agent_states)
            for idx, env in enumerate(self.pool):
                state = states[idx]
                action = actions[idx]
                history = histories[idx]
                next_state, r, is_done, is_tr, _ = env.step(action)
                cur_rewards[idx] += r
                cur_steps[idx] += 1
                history.append(Experience(state=state, action=action, reward=r, done_trunc=is_done or is_tr))
                if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                    yield tuple(history)
                states[idx] = next_state
                if is_done or is_tr:
                    # generate tail of history
                    if 0 < len(history) < self.steps_count:
                        yield tuple(history)
                    while len(history) > 1:
                        history.popleft()
                        yield tuple(history)
                    self.total_rewards.append(cur_rewards[idx])
                    self.total_steps.append(cur_steps[idx])
                    cur_rewards[idx] = 0.0
                    cur_steps[idx] = 0
                    states[idx], _ = self._reset_env(env)
                    self.agent_states[idx] = self.agent.initial_state()
                    history.clear()
            iter_idx += 1

    def pop_total_rewards(self) -> tt.List[float]:
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self) -> tt.List[tt.Tuple[float, int]]:
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


@dataclass(frozen=True)
class ExperienceFirstLast:
    state: State
    action: Action
    reward: float
    last_state: tt.Optional[State]


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env: gym.Env, agent: BaseAgent, gamma: float,
                 steps_count: int = 1, steps_delta: int = 1, env_seed: tt.Optional[int] = None):
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count+1, steps_delta, env_seed=env_seed)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self) -> tt.Generator[ExperienceFirstLast, None, None]:
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done_trunc and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)


def vector_rewards(rewards: tt.Deque[np.ndarray], dones: tt.Deque[np.ndarray], gamma: float) -> np.ndarray:
    """
    Calculate rewards from vectorized environment for given amount of steps.
    :param rewards: deque with observed rewards
    :param dones: deque with bool flags indicating end of episode
    :param gamma: gamma constant
    :return: vector with accumulated rewards
    """
    res = np.zeros(rewards[0].shape[0], dtype=np.float32)
    for r, d in zip(reversed(rewards), reversed(dones)):
        res *= gamma * (1. - d)
        res += r
    return res


class VectorExperienceSourceFirstLast(ExperienceSource):
    """
    ExperienceSourceFirstLast which supports VectorEnv from Gymnasium.
    """
    def __init__(self, env: gym.vector.VectorEnv, agent: BaseAgent,
                 gamma: float, steps_count: int = 1, env_seed: tt.Optional[int] = None,
                 unnest_data: bool = True):
        """
        Construct vectorized version of ExperienceSourceFirstLast
        :param env: vectorized environment
        :param agent: agent to use
        :param gamma: gamma for reward calculation
        :param steps_count: count of steps
        :param env_seed: seed for environments reset
        :param unnest_data: should we unnest data in the iterator. If True (default)
        ExperienceFirstLast will be yielded sequentially. If False, we'll keep them in a list as we
        got them from env vector.
        """
        super().__init__(env, agent, steps_count+1, steps_delta=1, env_seed=env_seed)
        self.env = env
        self.gamma = gamma
        self.steps = steps_count
        self.unnest_data = unnest_data
        self.agent_state = self.agent_states[0]

    def _iter_env_idx_obs_next(self, b_obs, b_next_obs) -> tt.Generator[tt.Tuple[
        int, tt.Any, tt.Any
    ], None, None]:
        """
        Iterate over individual environment observations and next observations.
        Take into account Tuple observation space (which is handled specially in Vectorized envs)
        :param b_obs: vectorized observations
        :param b_next_obs: vectorized next observations
        :yields: Tuple of index, observation and next observation
        """
        if isinstance(self.env.single_observation_space, gym.spaces.Tuple):
            obs_iter = zip(*b_obs)
            next_obs_iter = zip(*b_next_obs)
        else:
            obs_iter = b_obs
            next_obs_iter = b_next_obs
        yield from zip(range(self.env.num_envs), obs_iter, next_obs_iter)

    def __iter__(self) -> tt.Generator[tt.List[ExperienceFirstLast] | ExperienceFirstLast, None, None]:
        q_states = collections.deque(maxlen=self.steps+1)
        q_actions = collections.deque(maxlen=self.steps+1)
        q_rewards = collections.deque(maxlen=self.steps+1)
        q_dones = collections.deque(maxlen=self.steps+1)
        total_rewards = np.zeros(self.env.num_envs, dtype=np.float32)
        total_steps = np.zeros_like(total_rewards, dtype=np.int64)

        if self.env_seed is not None:
            seeds = [self.env_seed + idx for idx in range(self.env.num_envs)]
            self.env_seed += self.env.num_envs
            obs, _ = self.env.reset(seed=seeds)
        else:
            obs, _ = self.env.reset()
        env_indices = np.arange(self.env.num_envs)

        while True:
            q_states.append(obs)
            actions, self.agent_state = self.agent(obs, self.agent_state)
            q_actions.append(actions)
            next_obs, r, is_done, is_tr, _ = self.env.step(actions)
            total_rewards += r
            total_steps += 1
            done_or_tr = is_done | is_tr
            q_rewards.append(r)
            q_dones.append(done_or_tr)

            # process environments which are done at this step
            if done_or_tr.any():
                indices = env_indices[done_or_tr]
                self.total_steps.extend(total_steps[indices])
                self.total_rewards.extend(total_rewards[indices])
                total_steps[indices] = 0
                total_rewards[indices] = 0.0

            if len(q_states) == q_states.maxlen:
                # enough data for calculation
                results = []
                rewards = vector_rewards(q_rewards, q_dones, self.gamma)
                for i, e_obs, e_next_obs in self._iter_env_idx_obs_next(q_states[0], next_obs):
                    # if anywhere in the trajectory we have ended episode flag,
                    # the last state will be None
                    ep_ended = any(map(lambda d: d[i], q_dones))
                    last_state = e_next_obs if not ep_ended else None
                    results.append(ExperienceFirstLast(
                        state=e_obs,
                        action=q_actions[0][i],
                        reward=rewards[i],
                        last_state=last_state,
                    ))
                if self.unnest_data:
                    yield from results
                else:
                    yield results
            obs = next_obs


class ExperienceSourceBuffer:
    """
    The same as ExperienceSource, but takes episodes from the buffer
    """
    def __init__(self, buffer, steps_count=1):
        """
        Create buffered experience source
        :param buffer: list of episodes, each is a list of Experience object
        :param steps_count: count of steps in every entry
        """
        self.update_buffer(buffer)
        self.steps_count = steps_count

    def update_buffer(self, buffer):
        self.buffer = buffer
        self.lens = list(map(len, buffer))

    def __iter__(self):
        """
        Infinitely sample episode from the buffer and then sample item offset
        """
        while True:
            episode = random.randrange(len(self.buffer))
            ofs = random.randrange(self.lens[episode] - self.steps_count - 1)
            yield self.buffer[episode][ofs:ofs+self.steps_count]


class ExperienceReplayBuffer:
    def __init__(self, experience_source: tt.Optional[ExperienceSource], buffer_size: int):
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer: tt.List[ExperienceSource.Item] = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self) -> tt.Iterator[ExperienceSource.Item]:
        return iter(self.buffer)

    def sample(self, batch_size: int) -> tt.List[ExperienceSource.Item]:
        """
        Get one random batch from experience replay
        :param batch_size: size of the batch to sample
        :return: list of experience entries
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample: ExperienceSource.Item):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples: int):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)

    def state_dict(self) -> dict:
        return {
            "type": "uniform",
            "capacity": self.capacity,
            "pos": self.pos,
            "buffer": self.buffer,
        }

    def load_state_dict(self, state: dict):
        if state.get("capacity") != self.capacity:
            raise ValueError(
                f"Replay capacity mismatch: checkpoint={state.get('capacity')} current={self.capacity}"
            )
        loaded_buffer = list(state.get("buffer", []))
        if len(loaded_buffer) > self.capacity:
            loaded_buffer = loaded_buffer[-self.capacity:]
        self.buffer = loaded_buffer
        loaded_pos = int(state.get("pos", len(self.buffer)))
        if self.capacity <= 0:
            self.pos = 0
        else:
            self.pos = loaded_pos % self.capacity


class PrioReplayBufferNaive:
    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        max_prio = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = np.array(prios, dtype=np.float32) ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, experience_source, buffer_size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(experience_source, buffer_size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = utils.SumSegmentTree(it_capacity)
        self._it_min = utils.MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def _add(self, *args, **kwargs):
        idx = self.pos
        super()._add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in idxes]
        return samples, idxes, weights

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({
            "type": "prioritized",
            "alpha": self._alpha,
            "max_priority": self._max_priority,
            "sum_tree_values": np.array(self._it_sum._value, dtype=np.float32),
            "min_tree_values": np.array(self._it_min._value, dtype=np.float32),
        })
        return state

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)
        state_alpha = state.get("alpha")
        if state_alpha is not None and not np.isclose(float(state_alpha), float(self._alpha)):
            print(
                f"Warning: replay alpha mismatch, using current alpha={self._alpha} "
                f"instead of checkpoint alpha={state_alpha}"
            )
        self._max_priority = float(state.get("max_priority", 1.0))

        sum_values = state.get("sum_tree_values")
        min_values = state.get("min_tree_values")
        if sum_values is None or min_values is None:
            # Backfill trees for older snapshots that only stored transitions.
            for idx in range(len(self.buffer)):
                value = self._max_priority ** self._alpha
                self._it_sum[idx] = value
                self._it_min[idx] = value
            return

        if len(sum_values) != len(self._it_sum._value) or len(min_values) != len(self._it_min._value):
            raise ValueError("Segment tree size mismatch while loading prioritized replay buffer")
        self._it_sum._value = np.asarray(sum_values, dtype=np.float32).tolist()
        self._it_min._value = np.asarray(min_values, dtype=np.float32).tolist()


class BatchPreprocessor:
    """
    Abstract preprocessor class descendants to which converts experience
    batch to form suitable to learning.
    """
    def preprocess(self, batch):
        raise NotImplementedError


class QLearningPreprocessor(BatchPreprocessor):
    """
    Supports SimpleDQN, TargetDQN, DoubleDQN and can additionally feed TD-error back to
    experience replay buffer.

    To use different modes, use appropriate class method
    """
    def __init__(self, model, target_model, use_double_dqn=False, batch_td_error_hook=None, gamma=0.99, device="cpu"):
        self.model = model
        self.target_model = target_model
        self.use_double_dqn = use_double_dqn
        self.batch_dt_error_hook = batch_td_error_hook
        self.gamma = gamma
        self.device = device

    @staticmethod
    def simple_dqn(model, **kwargs):
        return QLearningPreprocessor(model=model, target_model=None, use_double_dqn=False, **kwargs)

    @staticmethod
    def target_dqn(model, target_model, **kwards):
        return QLearningPreprocessor(model, target_model, use_double_dqn=False, **kwards)

    @staticmethod
    def double_dqn(model, target_model, **kwargs):
        return QLearningPreprocessor(model, target_model, use_double_dqn=True, **kwargs)

    def _calc_Q(self, states_first, states_last):
        """
        Calculates apropriate q values for first and last states. Way of calculate depends on our settings.
        :param states_first: numpy array of first states
        :param states_last: numpy array of last states
        :return: tuple of numpy arrays of q values
        """
        # here we need both first and last values calculated using our main model, so we
        # combine both states into one batch for efficiency and separate results later
        if self.target_model is None or self.use_double_dqn:
            states_t = torch.tensor(np.concatenate((states_first, states_last), axis=0)).to(self.device)
            res_both = self.model(states_t).data.cpu().numpy()
            return res_both[:len(states_first)], res_both[len(states_first):]

        # in this case we have target_model set and use_double_dqn==False
        # so, we should calculate first_q and last_q using different models
        states_first_v = torch.tensor(states_first).to(self.device)
        states_last_v = torch.tensor(states_last).to(self.device)
        q_first = self.model(states_first_v).data
        q_last = self.target_model(states_last_v).data
        return q_first.cpu().numpy(), q_last.cpu().numpy()

    def _calc_target_rewards(self, states_last, q_last):
        """
        Calculate rewards from final states according to variants from our construction:
        1. simple DQN: max(Q(states, model))
        2. target DQN: max(Q(states, target_model))
        3. double DQN: Q(states, target_model)[argmax(Q(states, model)]
        :param states_last: numpy array of last states from the games
        :param q_last: numpy array of last q values
        :return: vector of target rewards
        """
        # in this case we handle both simple DQN and target DQN
        if self.target_model is None or not self.use_double_dqn:
            return q_last.max(axis=1)

        # here we have target_model set and use_double_dqn==True
        actions = q_last.argmax(axis=1)
        # calculate Q values using target net
        states_last_v = torch.tensor(states_last).to(self.device)
        q_last_target = self.target_model(states_last_v).data.cpu().numpy()
        return q_last_target[range(q_last_target.shape[0]), actions]

    def preprocess(self, batch):
        """
        Calculates data for Q learning from batch of observations
        :param batch: list of lists of Experience objects
        :return: tuple of numpy arrays:
            1. states -- observations
            2. target Q-values
            3. vector of td errors for every batch entry
        """
        # first and last states for every entry
        state_0 = np.array([exp[0].state for exp in batch], dtype=np.float32)
        state_L = np.array([exp[-1].state for exp in batch], dtype=np.float32)

        q0, qL = self._calc_Q(state_0, state_L)
        rewards = self._calc_target_rewards(state_L, qL)

        td = np.zeros(shape=(len(batch),))

        for idx, (total_reward, exps) in enumerate(zip(rewards, batch)):
            # game is done, no final reward
            if exps[-1].done:
                total_reward = 0.0
            for exp in reversed(exps[:-1]):
                total_reward *= self.gamma
                total_reward += exp.reward
            # update total reward and calculate td error
            act = exps[0].action
            td[idx] = q0[idx][act] - total_reward
            q0[idx][act] = total_reward

        return state_0, q0, td
