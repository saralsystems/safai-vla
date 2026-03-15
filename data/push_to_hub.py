"""Push LeRobot dataset to HuggingFace Hub."""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_CARD = """---
license: apache-2.0
task_categories:
  - robotics
tags:
  - safai-vla
  - lerobot
  - mujoco
  - manipulation
---

# safai-vla-mujoco-v0

Expert demonstration dataset for autonomous sewer maintenance robotics.

## Embodiment

- **Platform:** safai-vla-mujoco-proxy
- **Base:** Tracked differential-drive, 600mm width
- **Arm:** 4-DOF articulated arm with scoop end-effector
- **Cameras:** Front RGB (480x640), Wrist RGB (224x224)

## Action Space

7-DOF: `[base_fwd, base_lat, j1, j2, j3, j4, scoop]`

## Observation Keys

| Key | Shape | Type |
|-----|-------|------|
| front_rgb | (480, 640, 3) | uint8 |
| wrist_rgb | (224, 224, 3) | uint8 |
| joint_pos | (4,) | float32 |
| joint_vel | (4,) | float32 |
| ee_pos | (3,) | float32 |
| ee_quat | (4,) | float32 |
| base_pos | (3,) | float32 |

## Tasks

1. "navigate to blockage"
2. "assess and position for extraction"
3. "extract sludge at current position"
4. "deposit extracted material"

## Limitations

- Rigid-body sludge proxy (not particle fluid)
- MuJoCo Phase 0 environment (not full physics)
- Scripted expert policies (not human demonstrations)

## License

Apache 2.0
"""


def push_to_hub(dataset_path: str, repo_id: str) -> None:
    """Push dataset to HuggingFace Hub."""
    ds_path = Path(dataset_path)

    if not ds_path.exists():
        logger.error("Dataset path %s does not exist", dataset_path)
        return

    # Save dataset card
    card_path = ds_path / "README.md"
    card_path.write_text(DATASET_CARD)
    logger.info("Saved dataset card to %s", card_path)

    # Try to push
    try:
        from datasets import load_from_disk

        ds = load_from_disk(str(ds_path))
        ds.push_to_hub(repo_id)
        logger.info("Pushed dataset to https://huggingface.co/datasets/%s", repo_id)
    except ImportError:
        logger.warning("datasets library not available. Save dataset card locally.")
        logger.info("To push manually:")
        logger.info("  pip install datasets huggingface_hub")
        logger.info("  huggingface-cli login")
        logger.info(
            '  python -c "from datasets import load_from_disk; '
            "ds = load_from_disk('%s'); ds.push_to_hub('%s')\"",
            dataset_path,
            repo_id,
        )
    except Exception:
        logger.exception("Failed to push to hub. Dataset saved locally at %s", dataset_path)
        logger.info("To push manually, run:")
        logger.info("  huggingface-cli login")
        logger.info("  python -m data.push_to_hub --dataset %s --repo %s", dataset_path, repo_id)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Push dataset to HuggingFace Hub")
    parser.add_argument("--dataset", type=str, default="data/lerobot/")
    parser.add_argument("--repo", type=str, default="saralsystems/safai-vla-mujoco-v0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    push_to_hub(args.dataset, args.repo)


if __name__ == "__main__":
    main()
