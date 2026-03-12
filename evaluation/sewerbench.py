"""SewerBench -- evaluation harness for sewer-vla Phase 0.

Usage::

    python -m evaluation.sewerbench --checkpoint checkpoints/v0/ --episodes 100
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from envs.mujoco.sewer_env import TASKS, SewerVLAEnv
from evaluation.metrics import compute_task_metrics
from policies import POLICY_MAP

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


# ------------------------------------------------------------------
# Policy wrappers
# ------------------------------------------------------------------


class RandomPolicy:
    """Uniform random baseline that samples from the action space."""

    def __init__(self, action_dim: int = 7, seed: int | None = None):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------


def run_episode(
    env: SewerVLAEnv,
    policy: object,
    task: str,
    seed: int,
    inject_sludge_positions: bool = True,
) -> dict:
    """Run a single episode and return result metrics.

    Args:
        env: The MuJoCo sewer environment instance.
        policy: A callable ``policy(obs) -> action`` with a ``reset()`` method.
        task: Task language prompt string.
        seed: Seed for the episode reset.
        inject_sludge_positions: If True, add ``_sludge_positions`` to obs for
            scripted policies that need it.

    Returns:
        Dict with keys: task, success, steps, collision_count.
    """
    obs, info = env.reset(seed=seed, options={"task": task})
    policy.reset()

    # Provide set_env hook if the policy supports it
    if hasattr(policy, "set_env"):
        policy.set_env(env)

    total_collisions = 0
    for step_idx in range(env.config.max_episode_steps):
        if inject_sludge_positions:
            obs["_sludge_positions"] = env.get_sludge_positions()
        action = policy(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        total_collisions += info.get("collision_count", 0)
        if terminated or truncated:
            break

    return {
        "task": task,
        "success": bool(info.get("success", False)),
        "steps": step_idx + 1,
        "collision_count": total_collisions,
    }


# ------------------------------------------------------------------
# Evaluation driver
# ------------------------------------------------------------------


def evaluate_policy(
    policy_name: str,
    policy: object,
    episodes_per_task: int,
    base_seed: int = 0,
) -> dict:
    """Evaluate a single policy across all tasks.

    Returns:
        Dict mapping task name to its ``compute_task_metrics`` output, plus an
        ``"overall"`` key aggregating across tasks.
    """
    env = SewerVLAEnv()
    all_results: list[dict] = []
    per_task: dict[str, list[dict]] = {t: [] for t in TASKS}

    for task in TASKS:
        logger.info("  Task: %s", task)
        for ep in range(episodes_per_task):
            result = run_episode(
                env,
                policy,
                task,
                seed=base_seed + ep,
            )
            per_task[task].append(result)
            all_results.append(result)

    env.close()

    metrics: dict = {}
    for task in TASKS:
        metrics[task] = compute_task_metrics(per_task[task])
    metrics["overall"] = compute_task_metrics(all_results)
    metrics["policy"] = policy_name
    return metrics


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------


def save_bar_chart(report: dict, output_path: Path) -> None:
    """Save a bar chart of per-task success rates for each policy."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed -- skipping bar chart.")
        return

    policies = list(report.keys())
    tasks = TASKS
    x = np.arange(len(tasks))
    width = 0.8 / max(len(policies), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pname in enumerate(policies):
        rates = [report[pname].get(t, {}).get("success_rate", 0.0) * 100 for t in tasks]
        ax.bar(x + i * width, rates, width, label=pname)

    ax.set_ylabel("Success Rate (%)")
    ax.set_title("SewerBench v0 -- Per-Task Success Rates")
    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    short_labels = [t.split()[0] for t in tasks]
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=120)
    plt.close(fig)
    logger.info("Bar chart saved to %s", output_path)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SewerBench evaluation harness for sewer-vla Phase 0.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/v0/",
        help="Path to trained model checkpoint directory (optional).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes *per task* for each policy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "sewerbench_v0.json"),
        help="Path for JSON results file.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    report: dict[str, dict] = {}

    # ---- Random baseline ----
    logger.info("Evaluating: random policy")
    random_policy = RandomPolicy(seed=args.seed)
    report["random"] = evaluate_policy(
        "random",
        random_policy,
        args.episodes,
        base_seed=args.seed,
    )

    # ---- Scripted experts ----
    logger.info("Evaluating: scripted expert policies")
    for task, policy_cls in POLICY_MAP.items():
        policy = policy_cls(noise_scale=0.0, seed=args.seed)
        tag = f"expert_{task.split()[0]}"
        logger.info("  Policy: %s", tag)
        # Evaluate only on the matching task
        env = SewerVLAEnv()
        task_results: list[dict] = []
        for ep in range(args.episodes):
            result = run_episode(env, policy, task, seed=args.seed + ep)
            task_results.append(result)
        env.close()
        task_metrics = compute_task_metrics(task_results)
        task_metrics["policy"] = tag
        report[tag] = {task: task_metrics, "overall": task_metrics, "policy": tag}

    # ---- Trained checkpoint (optional) ----
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        logger.info("Checkpoint found at %s -- loading trained model.", ckpt_path)
        # TODO: Load SmolVLA + LoRA adapter and wrap as a callable policy.
        # For now, skip with a warning.
        logger.warning(
            "Trained model evaluation is not yet implemented. "
            "Add model loading logic once training pipeline is complete."
        )
    else:
        logger.info("No checkpoint at %s -- skipping trained model evaluation.", ckpt_path)

    # ---- Save results ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Results written to %s", output_path)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SewerBench v0 Results")
    print("=" * 60)
    for pname, pdata in report.items():
        overall = pdata.get("overall", {})
        sr = overall.get("success_rate", 0.0) * 100
        avg_steps = overall.get("avg_completion_time", 0.0)
        avg_col = overall.get("avg_collisions", 0.0)
        print(
            f"  {pname:30s}  success={sr:5.1f}%  steps={avg_steps:6.1f}  collisions={avg_col:5.1f}"
        )
    print("=" * 60 + "\n")

    # ---- Bar chart ----
    chart_path = output_path.with_suffix(".png")
    # Build a flat structure: policy -> task -> metrics for the chart
    chart_data: dict[str, dict] = {}
    for pname, pdata in report.items():
        chart_data[pname] = {t: pdata.get(t, {"success_rate": 0.0}) for t in TASKS}
    save_bar_chart(chart_data, chart_path)


if __name__ == "__main__":
    main()
