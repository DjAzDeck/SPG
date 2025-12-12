import os
import datetime
import torch
import numpy as np
import gymnasium as gym
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from multiple_versions import Trainer


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


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = resolve_device(cfg.experiment.device)
    time_steps = cfg.experiment.time_steps
    searches = cfg.experiment.searches
    if isinstance(searches, int):
        searches = [searches]
    run_ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for env_n in cfg.experiment.envs:
        for vers in cfg.experiment.versions:
            for exp_n in searches:
                for batches in cfg.experiment.batch_sizes:
                    env = gym.make(env_n)
                    test_env = gym.make(env_n)
                    env.reset(seed=cfg.experiment.seed)
                    test_env.reset(seed=cfg.experiment.seed)
                    env.action_space.seed(cfg.experiment.seed)
                    test_env.action_space.seed(cfg.experiment.seed)
                    torch.manual_seed(cfg.experiment.seed)
                    np.random.seed(cfg.experiment.seed)
                    state_dim = env.observation_space.shape[0]
                    action_dim = env.action_space.shape[0]
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
                    }
                    print(f"Benching architecture {vers} on environment {env_n}")
                    print(f"Searching {exp_n} times on batch size {batches} for {time_steps} timesteps")
                    print(f"Prioritized Replay Buffer: {cfg.experiment.prio}")
                    policy = Trainer(**trainer_kwargs)
                    name = f"{cfg.experiment.name}-{run_ts}-{vers}-{batches}-{exp_n}-{env_n}"
                    run_dir = os.path.join("runs", name)
                    persist_run_config(cfg, run_dir, meta={"env": env_n, "version": vers, "searches": exp_n, "batch_size": batches})
                    policy.train_routine(name, exp_n, time_steps, run_dir=run_dir)


if __name__ == "__main__":
    main()
