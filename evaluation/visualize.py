"""Episode replay visualization and metric plots."""

import argparse
import json
import logging
from pathlib import Path

import imageio

logger = logging.getLogger(__name__)


def render_episode_gif(
    task: str,
    seed: int,
    output_path: Path,
    max_frames: int = 100,
    fps: int = 10,
) -> None:
    """Render an episode as a GIF."""
    from envs.mujoco.sewer_env import SewerVLAEnv
    from policies import POLICY_MAP

    env = SewerVLAEnv()
    policy_cls = POLICY_MAP.get(task)
    if policy_cls is None:
        logger.error("Unknown task: %s", task)
        return

    policy = policy_cls(seed=seed)
    if hasattr(policy, "set_env"):
        policy.set_env(env)

    obs, info = env.reset(seed=seed, options={"task": task})
    frames = []

    for step in range(max_frames):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs["_sludge_positions"] = env.get_sludge_positions()
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    env.close()

    if frames:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
        logger.info("Saved %d-frame GIF to %s", len(frames), output_path)


def plot_results(results_path: str, output_path: str) -> None:
    """Generate bar chart of success rates from results JSON."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    with open(results_path) as f:
        results = json.load(f)

    tasks = []
    rates = []
    for policy_name, policy_results in results.items():
        for task, metrics in policy_results.items():
            tasks.append(f"{policy_name}\n{task[:20]}...")
            rates.append(metrics.get("success_rate", 0.0))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        range(len(tasks)), rates, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"] * 10
    )
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("SewerBench v0 Results")
    ax.set_ylim(0, 105)

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved plot to %s", output_path)


def main() -> None:
    """CLI entry point for visualization."""
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--results", type=str, default="evaluation/results/sewerbench_v0.json")
    parser.add_argument("--output-dir", type=str, default="evaluation/results/")
    parser.add_argument("--gifs", action="store_true", help="Generate sample episode GIFs")
    parser.add_argument("--num-gifs", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gifs:
        from envs.mujoco.sewer_env import TASKS

        gif_dir = output_dir / "episodes"
        gif_dir.mkdir(parents=True, exist_ok=True)

        for task in TASKS:
            for i in range(args.num_gifs):
                safe_task = task.replace(" ", "_")[:20]
                gif_path = gif_dir / f"{safe_task}_ep{i}.gif"
                render_episode_gif(task, seed=i, output_path=gif_path)

    if Path(args.results).exists():
        plot_path = output_dir / "sewerbench_v0.png"
        plot_results(args.results, str(plot_path))


if __name__ == "__main__":
    main()
