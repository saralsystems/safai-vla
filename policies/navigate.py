"""Scripted expert: navigate base toward nearest sludge blockage."""

import logging

import numpy as np

from policies.base import BasePolicy

logger = logging.getLogger(__name__)


class NavigatePolicy(BasePolicy):
    """Proportional controller that drives the base toward nearest sludge.

    Actions: [base_fwd, base_ang, 0, 0, 0, 0, 0] — arm stays stowed.
    Success: base within 0.3m of nearest sludge block.
    """

    def __init__(self, noise_scale: float = 0.0, seed: int | None = None):
        super().__init__(noise_scale=noise_scale, seed=seed)
        self._target_pos = None

    @property
    def task_name(self) -> str:
        return "navigate to blockage"

    def reset(self) -> None:
        super().reset()
        self._target_pos = None

    def _compute_action(self, obs: dict) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)

        if self._done:
            return action

        base_pos = obs["base_pos"]
        # Get sludge positions from the environment (passed in info or we use ee heuristic)
        # We use a simple heuristic: drive forward until close enough
        # The env reward is based on distance to nearest sludge

        # For scripted policy, we need access to sludge positions.
        # We'll pass them via obs if available, otherwise drive forward.
        sludge_pos = obs.get("_sludge_positions", None)

        if sludge_pos is not None and len(sludge_pos) > 0:
            # Find nearest sludge
            dists = np.linalg.norm(sludge_pos[:, :2] - base_pos[:2], axis=1)
            nearest_idx = np.argmin(dists)
            target = sludge_pos[nearest_idx]
            dist = dists[nearest_idx]

            if dist < 0.5:
                self._done = True
                return action

            # Direction to target
            dx = target[0] - base_pos[0]
            dy = target[1] - base_pos[1]

            # Forward: proportional with deceleration near target
            if abs(dx) > 0.5:
                action[0] = np.clip(np.sign(dx) * 1.0, -1.0, 1.0)
            else:
                action[0] = np.clip(dx * 2.0, -1.0, 1.0)

            # Lateral: proportional
            action[1] = np.clip(dy * 8.0, -1.0, 1.0)
        else:
            # Default: drive forward
            action[0] = 0.8

        return action


if __name__ == "__main__":
    import argparse

    from envs.mujoco.safai_env import SafaiVLAEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    if args.test:
        env = SafaiVLAEnv()
        policy = NavigatePolicy(seed=42)
        successes = 0
        for ep in range(args.episodes):
            obs, info = env.reset(seed=ep, options={"task": "navigate to blockage"})
            policy.reset()
            for step in range(500):
                obs["_sludge_positions"] = env.get_sludge_positions()
                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            if info.get("success", False):
                successes += 1
        env.close()
        rate = successes / args.episodes * 100
        print(f"NavigatePolicy: {successes}/{args.episodes} = {rate:.1f}% success")
