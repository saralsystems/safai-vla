"""Scripted expert: extract sludge by moving scoop to sludge, then lifting."""

import logging

import numpy as np

from policies.base import BasePolicy

logger = logging.getLogger(__name__)


class ExtractPolicy(BasePolicy):
    """Controller for sludge extraction using Jacobian IK.

    Phase 0: Move scoop to nearest sludge block (IK)
    Phase 1: Close scoop
    Phase 2: Lift arm

    Actions: [0, 0, j1, j2, j3, j4, scoop]
    Success: scoop within 0.15m of sludge block.
    """

    def __init__(self, noise_scale: float = 0.0, seed: int | None = None):
        super().__init__(noise_scale=noise_scale, seed=seed)
        self._phase = 0
        self._phase_step = 0
        self._env = None

    @property
    def task_name(self) -> str:
        return "extract sludge at current position"

    def set_env(self, env: object) -> None:
        """Set environment reference for Jacobian IK."""
        self._env = env

    def reset(self) -> None:
        super().reset()
        self._phase = 0
        self._phase_step = 0

    def _ik_action(self, target: np.ndarray) -> np.ndarray:
        """Compute arm joint actions via Jacobian IK toward target position."""
        action = np.zeros(7, dtype=np.float32)
        if self._env is None:
            return action

        import mujoco

        model = self._env.model
        data = self._env.data
        ee_site_id = self._env._ee_site_id

        ee_pos = data.site_xpos[ee_site_id]
        error = target - ee_pos

        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, ee_site_id)

        arm_jac_cols = []
        for jname in ["j1", "j2", "j3", "j4"]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            arm_jac_cols.append(model.jnt_dofadr[jid])

        J = jacp[:, arm_jac_cols]
        lam = 0.01
        JtJ = J.T @ J + lam * np.eye(4)
        dq = np.linalg.solve(JtJ, J.T @ error)

        scale = 10.0
        for i in range(4):
            action[2 + i] = np.clip(dq[i] * scale, -1.0, 1.0)

        return action

    def _compute_action(self, obs: dict) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)

        if self._done:
            return action

        self._phase_step += 1
        sludge_pos = obs.get("_sludge_positions", None)

        if self._phase == 0:
            # Phase 0: Move scoop toward nearest sludge with scoop open
            if sludge_pos is not None and len(sludge_pos) > 0:
                scoop_pos = obs["ee_pos"]
                dists = np.linalg.norm(sludge_pos - scoop_pos, axis=1)
                nearest = sludge_pos[np.argmin(dists)]
                target = nearest.copy()
                target[2] -= 0.02  # slightly below sludge for scooping

                action = self._ik_action(target)
                action[6] = 0.8  # open scoop

                if np.min(dists) < 0.12 or self._phase_step > 80:
                    self._phase = 1
                    self._phase_step = 0
            else:
                # Fallback: lower arm
                action[2] = -0.8
                action[3] = 0.6
                action[6] = 0.8
                if self._phase_step > 40:
                    self._phase = 1
                    self._phase_step = 0

        elif self._phase == 1:
            # Phase 1: Close scoop
            action[6] = -1.0
            if self._phase_step > 15:
                self._phase = 2
                self._phase_step = 0

        elif self._phase == 2:
            # Phase 2: Lift arm
            action[2] = 0.9
            action[3] = -0.5
            action[6] = -0.5  # keep closed
            if self._phase_step > 40:
                self._done = True

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
        policy = ExtractPolicy(seed=42)
        policy.set_env(env)
        successes = 0
        for ep in range(args.episodes):
            obs, info = env.reset(seed=ep, options={"task": "extract sludge at current position"})
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
        print(f"ExtractPolicy: {successes}/{args.episodes} = {rate:.1f}% success")
