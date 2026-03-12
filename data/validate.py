"""Dataset quality checks for raw and exported episodes."""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_KEYS = [
    "front_rgb",
    "wrist_rgb",
    "joint_pos",
    "joint_vel",
    "ee_pos",
    "ee_quat",
    "base_pos",
    "action",
    "reward",
    "timestamp",
]

EXPECTED_SHAPES = {
    "front_rgb": (None, 480, 640, 3),
    "wrist_rgb": (None, 224, 224, 3),
    "joint_pos": (None, 4),
    "joint_vel": (None, 4),
    "ee_pos": (None, 3),
    "ee_quat": (None, 4),
    "base_pos": (None, 3),
    "action": (None, 7),
}


def validate_episode(filepath: Path) -> list[str]:
    """Validate a single episode HDF5 file. Returns list of errors."""
    errors = []
    try:
        with h5py.File(filepath, "r") as f:
            for key in EXPECTED_KEYS:
                if key not in f:
                    errors.append(f"Missing dataset: {key}")
                    continue

                data = f[key]
                if key in EXPECTED_SHAPES:
                    expected = EXPECTED_SHAPES[key]
                    actual = data.shape
                    for i, (e, a) in enumerate(zip(expected, actual)):
                        if e is not None and e != a:
                            errors.append(
                                f"{key} shape mismatch at dim {i}: expected {e}, got {a}"
                            )

                if np.any(np.isnan(data[:])) and key != "front_rgb" and key != "wrist_rgb":
                    errors.append(f"{key} contains NaN values")

            # Check metadata
            for attr in ["task", "success", "episode_length"]:
                if attr not in f.attrs:
                    errors.append(f"Missing attribute: {attr}")

            # Check episode length consistency
            if "action" in f and "episode_length" in f.attrs:
                actual_len = f["action"].shape[0]
                claimed_len = f.attrs["episode_length"]
                if actual_len != claimed_len:
                    errors.append(
                        f"Length mismatch: action has {actual_len} steps, attrs says {claimed_len}"
                    )

    except Exception as e:
        errors.append(f"Failed to read file: {e}")

    return errors


def validate_dataset(input_dir: str) -> None:
    """Validate all episodes in a directory."""
    input_path = Path(input_dir)
    files = sorted(input_path.glob("*.h5"))

    if not files:
        logger.error("No .h5 files found in %s", input_dir)
        return

    total_errors = 0
    task_counts: dict[str, int] = {}
    success_counts: dict[str, int] = {}

    for filepath in files:
        errors = validate_episode(filepath)
        if errors:
            logger.warning("Errors in %s:", filepath.name)
            for err in errors:
                logger.warning("  - %s", err)
            total_errors += len(errors)
        else:
            with h5py.File(filepath, "r") as f:
                task = f.attrs.get("task", "unknown")
                success = f.attrs.get("success", False)
                task_counts[task] = task_counts.get(task, 0) + 1
                if success:
                    success_counts[task] = success_counts.get(task, 0) + 1

    logger.info("Validated %d episodes, %d errors", len(files), total_errors)
    logger.info("Task distribution:")
    for task, count in sorted(task_counts.items()):
        sc = success_counts.get(task, 0)
        logger.info(
            "  %s: %d episodes, %d successful (%.0f%%)",
            task,
            count,
            sc,
            100 * sc / count if count > 0 else 0,
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Validate episode dataset")
    parser.add_argument("--input", type=str, default="data/raw/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    validate_dataset(args.input)


if __name__ == "__main__":
    main()
