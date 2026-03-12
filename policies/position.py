"""Scripted expert: position arm end-effector above nearest sludge block."""

import logging

import numpy as np

from policies.base import BasePolicy

logger = logging.getLogger(__name__)


class PositionPolicy(BasePolicy):
    """Joint-space PD controller to move EE near nearest sludge.

    Uses MuJoCo Jacobian-based IK when env reference is available,
    otherwise falls back to heuristic joint mapping.

    Actions: [0, 0, j1, j2, j3, j4, 0] — base stationary, scoop unchanged.
    Success: EE within 0.2m of sludge block.
    """

    def __init__(self, noise_scale: float = 0.0, seed: int | None = None):
        super().__init__(noise_scale=noise_scale, seed=seed)
        self._target_pos = None
        self._env = None

    @property
    def task_name(self) -> str:
        return "assess and position for extraction"

    def set_env(self, env: object) -> None:
        """Set environment reference for Jacobian-based IK."""
        self._env = env

    def reset(self) -> None:
        super().reset()
        self._target_pos = None

    def _compute_action(self, obs: dict) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)

        if self._done:
            return action

        ee_pos = obs["ee_pos"]
        sludge_pos = obs.get("_sludge_positions", None)

        if sludge_pos is None or len(sludge_pos) == 0:
            return action

        # Find nearest sludge and target slightly above it
        dists = np.linalg.norm(sludge_pos - ee_pos, axis=1)
        nearest_idx = np.argmin(dists)
        target = sludge_pos[nearest_idx].copy()
        target[2] += 0.05  # slightly above sludge surface

        dist = np.linalg.norm(target - ee_pos)

        if dist < 0.15:
            self._done = True
            return action

        error = target - ee_pos

        # Use Jacobian-based IK if env is available
        if self._env is not None:
            import mujoco

            model = self._env.model
            data = self._env.data
            ee_site_id = self._env._ee_site_id

            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, None, ee_site_id)

            # Extract columns for arm joints (j1-j4)
            # Joint DOF addresses for j1-j4
            arm_jac_cols = []
            for jname in ["j1", "j2", "j3", "j4"]:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                dof_adr = model.jnt_dofadr[jid]
                arm_jac_cols.append(dof_adr)

            J = jacp[:, arm_jac_cols]  # 3x4 Jacobian for arm joints

            # Damped pseudoinverse
            lam = 0.01
            JtJ = J.T @ J + lam * np.eye(4)
            dq = np.linalg.solve(JtJ, J.T @ error)

            # Scale to action range [-1, 1]
            # dq is in radians, action * 0.1 = joint delta per step
            scale = 10.0  # inverse of 0.1 step scaling
            for i in range(4):
                action[2 + i] = np.clip(dq[i] * scale, -1.0, 1.0)
        else:
            # Fallback heuristic
            gain = 5.0
            action[2] = np.clip(-error[2] * gain, -1.0, 1.0)
            action[3] = np.clip(error[0] * gain, -1.0, 1.0)
            action[4] = np.clip((error[0] + error[2]) * gain * 0.5, -1.0, 1.0)
            action[5] = np.clip(-error[2] * gain * 0.3, -1.0, 1.0)

        return action


if __name__ == "__main__":
    import argparse

    from envs.mujoco.sewer_env import SewerVLAEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    if args.test:
        env = SewerVLAEnv()
        policy = PositionPolicy(seed=42)
        policy.set_env(env)
        successes = 0
        for ep in range(args.episodes):
            obs, info = env.reset(seed=ep, options={"task": "assess and position for extraction"})
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
        print(f"PositionPolicy: {successes}/{args.episodes} = {rate:.1f}% success")
