import os
import datetime
from typing import Any
import torch
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from multiple_versions import Trainer
from lib.env_wrappers import get_env_dimensions, make_env, make_vector_env


def resolve_device(preference: str) -> torch.device:
    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            print("Using CUDA device")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            print("Using MPS device")
            return torch.device("mps")
        print("Using CPU device (auto fallback)")
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        print("Using CUDA device")
        return torch.device("cuda")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        print("Using MPS device")
        return torch.device("mps")
    print("Using CPU device (explicit)")
    return torch.device("cpu")


def persist_run_config(cfg: DictConfig, run_dir: str, meta: dict | None = None):
    os.makedirs(run_dir, exist_ok=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg_dict, dict):
        cfg_dict["run_metadata"] = meta or {}
    OmegaConf.save(config=OmegaConf.create(cfg_dict), f=os.path.join(run_dir, "config.yaml"))
    try:
        overrides = HydraConfig.get().overrides.task
        if overrides:
            with open(os.path.join(run_dir, "overrides.txt"), "w") as f:
                f.write("\n".join(overrides))
    except Exception:
        pass


def _normalize_env_spec(env_spec: Any) -> Any:
    if isinstance(env_spec, DictConfig):
        return OmegaConf.to_container(env_spec, resolve=True)
    return env_spec


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = resolve_device(cfg.experiment.device)
    time_steps = cfg.experiment.time_steps
    searches = cfg.experiment.searches
    if isinstance(searches, int):
        searches = [searches]
    run_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for env_spec in cfg.experiment.envs:
        env_obj = _normalize_env_spec(env_spec)
        env_id = env_obj["id"] if isinstance(env_obj, dict) else env_obj
        for vers in cfg.experiment.versions:
            for exp_n in searches:
                for batches in cfg.experiment.batch_sizes:
                    num_envs = int(cfg.trainer.parallel_envs.num_envs)
                    vector_mode = str(cfg.trainer.parallel_envs.mode).lower()
                    if vector_mode not in {"sync", "async"}:
                        raise ValueError("trainer.parallel_envs.mode must be one of: sync, async")

                    env = None
                    test_env = None
                    shape_probe_env = None
                    try:
                        if num_envs > 1:
                            env = make_vector_env(
                                env_obj,
                                seed=cfg.experiment.seed,
                                num_envs=num_envs,
                                asynchronous=(vector_mode == "async"),
                            )
                        else:
                            env = make_env(env_obj, seed=cfg.experiment.seed)
                        shape_probe_env = make_env(env_obj, seed=cfg.experiment.seed + 11)
                        test_env = make_env(env_obj, seed=cfg.experiment.seed + 1)

                        torch.manual_seed(cfg.experiment.seed)
                        np.random.seed(cfg.experiment.seed)
                        state_dim, action_dim = get_env_dimensions(shape_probe_env)

                        trainer_kwargs = {
                            "env": env,
                            "test_env": test_env,
                            "state_dim": state_dim,
                            "action_dim": action_dim,
                            "device": device,
                            "version": vers,
                            "batch_size": batches,
                            "prio": cfg.experiment.prio,
                            "seed": cfg.experiment.seed,
                            "replay_size": cfg.trainer.replay_size,
                            "discount": cfg.trainer.discount,
                            "LR": cfg.trainer.lr,
                            "prio_alpha": cfg.trainer.prioritized_replay.alpha,
                            "rollout_step": cfg.trainer.rollout_step,
                            "policy_freq": cfg.trainer.policy_freq,
                            "noise_clip": cfg.trainer.noise_clip,
                            "policy_noise": cfg.trainer.policy_noise,
                            "alpha": cfg.trainer.alpha,
                            "tau": cfg.trainer.tau,
                            "sigma_start": cfg.trainer.sigma.start,
                            "sigma_final": cfg.trainer.sigma.final,
                            "sigma_decay_last_frame": cfg.trainer.sigma.decay_last_frame,
                            "beta_start": cfg.trainer.prioritized_replay.beta_start,
                            "beta_frames": cfg.trainer.prioritized_replay.beta_frames,
                            "replay_initial": cfg.trainer.replay_initial,
                            "test_iters": cfg.trainer.test_iters,
                            "num_envs": num_envs,
                            "checkpoint_interval": cfg.trainer.checkpoint.save_interval,
                            "checkpoint_load_path": cfg.trainer.checkpoint.load_path,
                            "resume_from_checkpoint": cfg.trainer.checkpoint.resume,
                            "save_replay_buffer": cfg.trainer.replay_buffer.save,
                            "replay_save_interval": cfg.trainer.replay_buffer.save_interval,
                            "replay_load_path": cfg.trainer.replay_buffer.load_path,
                            "search_final": cfg.trainer.search.final,
                            "search_decay_last_frame": cfg.trainer.search.decay_last_frame,
                            "search_chunk_size": cfg.trainer.search.chunk_size,
                            "summary_interval_sec": cfg.trainer.logging.summary_interval_sec,
                            "tqdm_enabled": cfg.trainer.logging.tqdm,
                            "eval_episodes": cfg.trainer.logging.eval_episodes,
                        }
                        print(f"Benching architecture {vers} on environment {env_id}")
                        print(f"Searching {exp_n} times on batch size {batches} for {time_steps} timesteps")
                        print(f"Prioritized Replay Buffer: {cfg.experiment.prio}")
                        print(f"Parallel environments: {num_envs} ({vector_mode})")
                        policy = Trainer(**trainer_kwargs)
                        tags = [f"envs{num_envs}-{vector_mode if num_envs > 1 else 'single'}"]
                        if cfg.trainer.checkpoint.load_path:
                            tags.append("ckpt-load")
                        if cfg.trainer.checkpoint.resume:
                            tags.append("resume")
                        if cfg.trainer.replay_buffer.load_path:
                            tags.append("replay-load")
                        name = f"{cfg.experiment.name}-{run_ts}-{vers}-{batches}-{exp_n}-{env_id}-{'-'.join(tags)}"
                        run_dir = os.path.join("runs", name)
                        persist_run_config(
                            cfg,
                            run_dir,
                            meta={
                                "env": env_spec,
                                "version": vers,
                                "searches": exp_n,
                                "batch_size": batches,
                                "tags": tags,
                                "num_envs": num_envs,
                                "vector_mode": vector_mode,
                                "checkpoint_load_path": cfg.trainer.checkpoint.load_path,
                                "checkpoint_resume": bool(cfg.trainer.checkpoint.resume),
                                "replay_load_path": cfg.trainer.replay_buffer.load_path,
                            },
                        )
                        policy.train_routine(name, exp_n, time_steps, run_dir=run_dir, tags=tags)
                    finally:
                        if shape_probe_env is not None:
                            shape_probe_env.close()
                        if env is not None:
                            env.close()
                        if test_env is not None:
                            test_env.close()


if __name__ == "__main__":
    main()
