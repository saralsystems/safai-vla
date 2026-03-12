"""Metric computation functions for SewerBench evaluation."""

import logging

logger = logging.getLogger(__name__)


def compute_success_rate(results: list[dict]) -> float:
    """Compute fraction of episodes that succeeded.

    Args:
        results: List of episode result dicts, each with a ``success`` bool key.

    Returns:
        Success rate in [0, 1]. Returns 0.0 if *results* is empty.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("success", False)) / len(results)


def compute_avg_completion_time(results: list[dict]) -> float:
    """Compute mean episode length (steps) across results.

    Args:
        results: List of episode result dicts, each with a ``steps`` int key.

    Returns:
        Mean number of steps. Returns 0.0 if *results* is empty.
    """
    if not results:
        return 0.0
    return sum(r.get("steps", 0) for r in results) / len(results)


def compute_avg_collisions(results: list[dict]) -> float:
    """Compute mean collision count (robot-pipe contacts) across results.

    Args:
        results: List of episode result dicts, each with a ``collision_count`` int key.

    Returns:
        Mean collision count. Returns 0.0 if *results* is empty.
    """
    if not results:
        return 0.0
    return sum(r.get("collision_count", 0) for r in results) / len(results)


def compute_task_metrics(results: list[dict]) -> dict:
    """Aggregate all metrics for a set of episode results.

    Args:
        results: List of episode result dicts. Each dict must contain:
            - task (str): task name
            - success (bool): whether the episode succeeded
            - steps (int): number of steps taken
            - collision_count (int): robot-pipe collision count

    Returns:
        Dictionary with keys ``success_rate``, ``avg_completion_time``,
        ``avg_collisions``, and ``num_episodes``.
    """
    return {
        "success_rate": compute_success_rate(results),
        "avg_completion_time": compute_avg_completion_time(results),
        "avg_collisions": compute_avg_collisions(results),
        "num_episodes": len(results),
    }
