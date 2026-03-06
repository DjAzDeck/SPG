#!/usr/bin/env python3
import argparse
import importlib.util
import numpy as np
import torch
import gymnasium as gym
from tensorboardX import SummaryWriter

import ptan
from lib import spg_torch
from lib.env_wrappers import get_env_dimensions, make_env

ENV_ID = "HalfCheetah-v5"


def load_actor_state_dict(model_path: str, device: torch.device) -> dict:
    payload = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "actor" in payload:
        return payload["actor"]
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model or checkpoint file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help=f"Environment name to use, default={ENV_ID}")
    parser.add_argument("-r", "--record-dir", required=False, help="If specified, save evaluation videos to this directory")
    parser.add_argument("-n", "--name", required=True, help="Name of the test run")
    parser.add_argument("-k", "--kappa", required=False, default=10, type=int, help="Number of evaluation episodes")
    parser.add_argument("-s", "--seed", required=False, default=0, type=int)
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for evaluation")
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["human", "rgb_array", "none"],
        default="human",
        help="Gymnasium render mode. Recording uses rgb_array.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=None, help="Optional time limit wrapper for evaluation")

    args = parser.parse_args()
    device = torch.device(args.device)
    render_mode = None if args.render_mode == "none" else args.render_mode
    if args.record_dir:
        render_mode = "rgb_array"

    env_spec = {"id": args.env}
    if render_mode is not None:
        env_spec["render_mode"] = render_mode
    if args.max_episode_steps is not None:
        env_spec["max_episode_steps"] = args.max_episode_steps

    env = make_env(env_spec, seed=args.seed)
    if args.record_dir:
        if importlib.util.find_spec("moviepy") is None:
            env.close()
            raise SystemExit(
                "Video recording requires MoviePy. Install it with "
                '`uv add "gymnasium[other]"` or `uv add moviepy`.'
            )
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.record_dir,
            episode_trigger=lambda _: True,
            name_prefix=args.name,
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    writer = SummaryWriter(comment="-spg_" + args.name)
    state_dim, action_dim = get_env_dimensions(env)
    net = spg_torch.SPGActor(state_dim, action_dim).to(device)
    net.load_state_dict(load_actor_state_dict(args.model, device))
    net.eval()
    full_reward = []

    try:
        for frame_idx in range(args.kappa):
            obs, _ = env.reset(seed=args.seed + frame_idx)
            total_reward = 0.0
            total_steps = 0
            while True:
                if render_mode == "human":
                    env.render()
                obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
                with torch.no_grad():
                    mu_v = net(obs_v)
                action = mu_v.squeeze(dim=0).cpu().numpy()
                action = np.clip(action, -1.0, 1.0)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                total_steps += 1
                if terminated or truncated:
                    break
            print("In %d steps we got %.3f reward" % (total_steps, total_reward))
            full_reward.append(total_reward)
            writer.add_scalar("test_reward", total_reward, frame_idx)
            writer.add_scalar("average_test_reward", np.mean(full_reward), frame_idx)
    finally:
        writer.close()
        env.close()

    print("Reward statistics after %d runs: \n -------------" % args.kappa)
    print(
        "Average reward: %.3f \n \
    Std reward: %.3f \n \
    Max reward: %.3f"
        % (np.mean(full_reward), np.std(full_reward), np.max(full_reward))
    )
