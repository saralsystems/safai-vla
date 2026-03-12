"""Convert raw HDF5 episodes to a LeRobot-compatible HuggingFace dataset.

Works standalone without the lerobot library. Requires only h5py, datasets,
numpy, and PIL.

Usage:
    python -m data.export_lerobot --input data/raw/ --output data/lerobot/
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# Column names that mirror LeRobot conventions.
STATE_KEYS = ["joint_pos", "joint_vel", "ee_pos", "ee_quat", "base_pos"]
STATE_DIMS = {"joint_pos": 4, "joint_vel": 4, "ee_pos": 3, "ee_quat": 4, "base_pos": 3}
ACTION_DIM = 7


def _build_features() -> Features:
    """Define the Arrow schema for the exported dataset."""
    return Features(
        {
            "observation.images.front": Image(),
            "observation.images.wrist": Image(),
            "observation.state": Sequence(Value("float32"), length=18),
            "action": Sequence(Value("float32"), length=ACTION_DIM),
            "reward": Value("float64"),
            "timestamp": Value("float64"),
            "episode_index": Value("int64"),
            "frame_index": Value("int64"),
            "task": Value("string"),
            "success": Value("bool"),
            "next.reward": Value("float64"),
            "next.done": Value("bool"),
            "index": Value("int64"),
        }
    )


def _read_episode(filepath: Path) -> dict:
    """Read a single HDF5 episode file and return arrays + metadata."""
    with h5py.File(filepath, "r") as f:
        data = {
            key: f[key][:]
            for key in [*STATE_KEYS, "front_rgb", "wrist_rgb", "action", "reward", "timestamp"]
        }
        meta = {
            "task": f.attrs["task"],
            "success": bool(f.attrs["success"]),
            "episode_length": int(f.attrs["episode_length"]),
        }
    return data, meta


def _concat_state(data: dict) -> np.ndarray:
    """Concatenate state arrays into a flat (T, 18) vector."""
    arrays = [data[k] for k in STATE_KEYS]
    return np.concatenate(arrays, axis=1).astype(np.float32)


def _numpy_to_pil(arr: np.ndarray) -> PILImage.Image:
    """Convert a uint8 HWC numpy array to a PIL Image."""
    return PILImage.fromarray(arr)


def export_episodes(input_dir: Path, output_dir: Path) -> None:
    """Read all HDF5 episodes and write a HuggingFace Arrow dataset."""
    episode_files = sorted(input_dir.glob("episode_*.h5"))
    if not episode_files:
        logger.error("No episode HDF5 files found in %s", input_dir)
        return

    logger.info("Found %d episode files in %s", len(episode_files), input_dir)

    rows: list[dict] = []
    global_idx = 0
    episode_lengths: list[int] = []
    task_counts: dict[str, int] = {}

    for ep_idx, ep_file in enumerate(episode_files):
        data, meta = _read_episode(ep_file)
        T = meta["episode_length"]
        episode_lengths.append(T)
        task_counts[meta["task"]] = task_counts.get(meta["task"], 0) + 1

        state = _concat_state(data)
        actions = data["action"].astype(np.float32)
        rewards = data["reward"]
        timestamps = data["timestamp"]

        for t in range(T):
            next_done = t == T - 1
            next_reward = rewards[t + 1] if t + 1 < T else 0.0

            rows.append(
                {
                    "observation.images.front": _numpy_to_pil(data["front_rgb"][t]),
                    "observation.images.wrist": _numpy_to_pil(data["wrist_rgb"][t]),
                    "observation.state": state[t].tolist(),
                    "action": actions[t].tolist(),
                    "reward": float(rewards[t]),
                    "timestamp": float(timestamps[t]),
                    "episode_index": ep_idx,
                    "frame_index": t,
                    "task": meta["task"],
                    "success": meta["success"],
                    "next.reward": float(next_reward),
                    "next.done": next_done,
                    "index": global_idx,
                }
            )
            global_idx += 1

        if (ep_idx + 1) % 50 == 0:
            logger.info(
                "Processed %d / %d episodes (%d frames)",
                ep_idx + 1,
                len(episode_files),
                global_idx,
            )

    logger.info("Total: %d episodes, %d frames", len(episode_files), global_idx)

    features = _build_features()
    dataset = Dataset.from_list(rows, features=features)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir / "dataset"))
    logger.info("Saved Arrow dataset to %s", output_dir / "dataset")

    # Write metadata sidecar for downstream consumers.
    meta_info = {
        "robot_type": "sewer-vla-mujoco-proxy",
        "total_episodes": len(episode_files),
        "total_frames": global_idx,
        "fps": 20,
        "action_dim": ACTION_DIM,
        "state_dim": 18,
        "state_keys": STATE_KEYS,
        "state_dims": STATE_DIMS,
        "task_counts": task_counts,
        "episode_lengths": episode_lengths,
        "license": "Apache-2.0",
    }
    meta_path = output_dir / "meta" / "info.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta_info, indent=2))
    logger.info("Wrote dataset metadata to %s", meta_path)

    # Write per-episode metadata (episode_index -> length mapping).
    episodes_meta = [
        {"episode_index": i, "length": episode_lengths[i]} for i in range(len(episode_lengths))
    ]
    ep_meta_path = output_dir / "meta" / "episodes.json"
    ep_meta_path.write_text(json.dumps(episodes_meta, indent=2))
    logger.info("Wrote episode metadata to %s", ep_meta_path)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export raw HDF5 episodes to LeRobot-compatible HuggingFace dataset"
    )
    parser.add_argument(
        "--input", type=str, default="data/raw/", help="Directory with episode_*.h5 files"
    )
    parser.add_argument(
        "--output", type=str, default="data/lerobot/", help="Output directory for the dataset"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    export_episodes(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
