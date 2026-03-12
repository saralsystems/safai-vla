"""Collect expert demonstrations by running scripted policies in MuJoCo env."""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from envs.mujoco.sewer_env import TASKS, SewerVLAEnv
from policies import POLICY_MAP

logger = logging.getLogger(__name__)


def collect_episode(
    env: SewerVLAEnv,
    task: str,
    noise_scale: float,
    seed: int,
    max_steps: int = 500,
) -> dict:
    """Run one episode and record all data."""
    policy_cls = POLICY_MAP[task]
    policy = policy_cls(noise_scale=noise_scale, seed=seed)
    if hasattr(policy, "set_env"):
        policy.set_env(env)

    obs, info = env.reset(seed=seed, options={"task": task})

    # Pre-allocate arrays
    front_rgbs = []
    wrist_rgbs = []
    joint_positions = []
    joint_velocities = []
    ee_positions = []
    ee_quats = []
    base_positions = []
    actions = []
    rewards = []
    timestamps = []

    for step in range(max_steps):
        obs["_sludge_positions"] = env.get_sludge_positions()
        action = policy(obs)

        # Record observation + action
        front_rgbs.append(obs["front_rgb"])
        wrist_rgbs.append(obs["wrist_rgb"])
        joint_positions.append(obs["joint_pos"].copy())
        joint_velocities.append(obs["joint_vel"].copy())
        ee_positions.append(obs["ee_pos"].copy())
        ee_quats.append(obs["ee_quat"].copy())
        base_positions.append(obs["base_pos"].copy())
        actions.append(action.copy())
        timestamps.append(step / env.config.control_frequency_hz)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    episode = {
        "front_rgb": np.array(front_rgbs),
        "wrist_rgb": np.array(wrist_rgbs),
        "joint_pos": np.array(joint_positions),
        "joint_vel": np.array(joint_velocities),
        "ee_pos": np.array(ee_positions),
        "ee_quat": np.array(ee_quats),
        "base_pos": np.array(base_positions),
        "action": np.array(actions),
        "reward": np.array(rewards),
        "timestamp": np.array(timestamps),
        "task": task,
        "success": info.get("success", False),
        "episode_length": len(actions),
        "noise_scale": noise_scale,
        "seed": seed,
    }
    return episode


def save_episode_hdf5(episode: dict, filepath: Path) -> None:
    """Save episode data as HDF5."""
    with h5py.File(filepath, "w") as f:
        # Image data (compressed)
        f.create_dataset(
            "front_rgb", data=episode["front_rgb"], compression="gzip", compression_opts=4
        )
        f.create_dataset(
            "wrist_rgb", data=episode["wrist_rgb"], compression="gzip", compression_opts=4
        )

        # State data
        f.create_dataset("joint_pos", data=episode["joint_pos"])
        f.create_dataset("joint_vel", data=episode["joint_vel"])
        f.create_dataset("ee_pos", data=episode["ee_pos"])
        f.create_dataset("ee_quat", data=episode["ee_quat"])
        f.create_dataset("base_pos", data=episode["base_pos"])

        # Action and reward
        f.create_dataset("action", data=episode["action"])
        f.create_dataset("reward", data=episode["reward"])
        f.create_dataset("timestamp", data=episode["timestamp"])

        # Metadata
        f.attrs["task"] = episode["task"]
        f.attrs["success"] = episode["success"]
        f.attrs["episode_length"] = episode["episode_length"]
        f.attrs["noise_scale"] = episode["noise_scale"]
        f.attrs["seed"] = episode["seed"]


def collect_all(
    output_dir: str,
    tasks: list[str],
    episodes_per_task: int,
    noisy_ratio: float = 0.7,
    noise_scale: float = 0.1,
    base_seed: int = 0,
) -> None:
    """Collect episodes across all tasks."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env = SewerVLAEnv()
    episode_idx = 0

    for task in tasks:
        logger.info("Collecting %d episodes for task: %s", episodes_per_task, task)
        for i in tqdm(range(episodes_per_task), desc=task[:20]):
            seed = base_seed + episode_idx
            use_noise = i < int(episodes_per_task * noisy_ratio)
            ns = noise_scale if use_noise else 0.0

            episode = collect_episode(env, task, ns, seed)

            ep_file = output_path / f"episode_{episode_idx:05d}.h5"
            save_episode_hdf5(episode, ep_file)

            episode_idx += 1

    env.close()
    logger.info("Collected %d episodes total in %s", episode_idx, output_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Collect expert demonstrations")
    parser.add_argument("--task", type=str, default="all", help="Task name or 'all'")
    parser.add_argument("--episodes", type=int, default=500, help="Total episodes")
    parser.add_argument("--output", type=str, default="data/raw/", help="Output directory")
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--noisy-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.task == "all":
        tasks = TASKS
        episodes_per_task = args.episodes // len(tasks)
    else:
        tasks = [args.task]
        episodes_per_task = args.episodes

    collect_all(
        output_dir=args.output,
        tasks=tasks,
        episodes_per_task=episodes_per_task,
        noisy_ratio=args.noisy_ratio,
        noise_scale=args.noise_scale,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
