import copy
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf


class ActionScaleWrapper(gym.ActionWrapper):
    """
    Rescale Box actions from the model-facing [-1, 1] range back to the
    environment's original action bounds.
    """

    def __init__(self, env: gym.Env):
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("ActionScaleWrapper only supports Box action spaces.")
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("Action bounds must be finite to apply scaling.")
        span = high - low
        if np.any(span <= 0):
            raise ValueError("Action bounds must satisfy low < high for every dimension.")

        super().__init__(env)
        self._orig_low = low
        self._orig_high = high
        self._orig_span = span
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        scaled = self._orig_low + (clipped + 1.0) * 0.5 * self._orig_span
        return scaled.astype(np.float32, copy=False)

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        orig_action = np.asarray(action, dtype=np.float32)
        return 2.0 * (orig_action - self._orig_low) / self._orig_span - 1.0


def _to_plain_dict(env_spec: Any) -> Any:
    if isinstance(env_spec, DictConfig):
        return OmegaConf.to_container(env_spec, resolve=True)
    return env_spec


def _parse_env_spec(env_spec: Any) -> tuple[str, dict[str, Any]]:
    plain_spec = _to_plain_dict(env_spec)
    if isinstance(plain_spec, str):
        return plain_spec, {}
    if isinstance(plain_spec, dict):
        env_id = plain_spec.get("id")
        if not env_id:
            raise ValueError("env_spec dict requires a non-empty 'id' field.")
        params = copy.deepcopy(plain_spec)
        params.pop("id", None)
        return env_id, params
    raise TypeError("env_spec must be a Gymnasium id string or a mapping with an 'id' key.")


def _box_space_dim(space: gym.Space, name: str) -> int:
    if not isinstance(space, spaces.Box):
        raise TypeError(
            f"{name} space must be gymnasium.spaces.Box. "
            f"Discrete environments are not supported, got {type(space).__name__}."
        )
    if len(space.shape) != 1:
        raise TypeError(f"{name} space must be one-dimensional, got shape={space.shape}.")
    return int(space.shape[0])


def get_env_dimensions(env: gym.Env) -> tuple[int, int]:
    """
    Validate the environment contract expected by the actor/critic stack and
    return flat observation/action dimensions.
    """

    state_dim = _box_space_dim(env.observation_space, "Observation")
    action_dim = _box_space_dim(env.action_space, "Action")
    return state_dim, action_dim


def make_env(env_spec: Any, seed: int | None = None) -> gym.Env:
    """
    Create a continuous-control Gymnasium environment from a string id or a
    mapping like {"id": "HalfCheetah-v5", "render_mode": "rgb_array"}.
    """

    env_id, params = _parse_env_spec(env_spec)
    max_episode_steps = params.pop("max_episode_steps", None)

    env = gym.make(env_id, **params)
    try:
        get_env_dimensions(env)
        env = ActionScaleWrapper(env)
        get_env_dimensions(env)
        if max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=int(max_episode_steps))
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return env
    except Exception:
        env.close()
        raise


def make_env_factory(env_spec: Any, seed: int | None = None) -> Callable[[], gym.Env]:
    plain_spec = _to_plain_dict(env_spec)

    def _factory() -> gym.Env:
        return make_env(plain_spec, seed=seed)

    return _factory


def make_vector_env(
    env_spec: Any,
    seed: int,
    num_envs: int,
    asynchronous: bool = True,
) -> gym.vector.VectorEnv:
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    env_fns = [make_env_factory(env_spec, seed + idx) for idx in range(num_envs)]
    if asynchronous and num_envs > 1:
        env = gym.vector.AsyncVectorEnv(env_fns)
    else:
        env = gym.vector.SyncVectorEnv(env_fns)

    env.reset(seed=[seed + idx for idx in range(num_envs)])
    return env
