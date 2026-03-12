"""Scripted expert: move arm to deposit zone and release sludge."""

import logging

import numpy as np

from policies.base import BasePolicy

logger = logging.getLogger(__name__)


class DepositPolicy(BasePolicy):
    """Controller to move scoop toward deposit zone, then open.

    Phase 0: Move EE toward deposit zone using Jacobian IK
    Phase 1: Open scoop
    Phase 2: Return to stowed position

    Actions: [0, 0, j1, j2, j3, j4, scoop]
    Success: scoop within 0.5m of deposit zone.
    """

    def __init__(self, noise_scale: float = 0.0, seed: int | None = None):
        super().__init__(noise_scale=noise_scale, seed=seed)
        self._phase = 0
        self._phase_step = 0
        self._env = None

    @property
    def task_name(self) -> str:
        return "deposit extracted material"

    def set_env(self, env: object) -> None:
        """Set environment reference for Jacobian IK."""
        self._env = env

    def reset(self) -> None:
        super().reset()
        self._phase = 0
        self._phase_step = 0

    def _compute_action(self, obs: dict) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)

        if self._done:
            return action

        self._phase_step += 1

        if self._phase == 0:
            # Phase 0: Move EE toward deposit zone
            if self._env is not None:
                import mujoco

                model = self._env.model
                data = self._env.data

                deposit_pos = self._env.get_deposit_position()
                target = deposit_pos.copy()
                target[2] = 0.15  # above deposit zone

                ee_pos = data.site_xpos[self._env._ee_site_id]
                error = target - ee_pos
                dist = np.linalg.norm(error)

                if dist < 0.3:
                    self._phase = 1
                    self._phase_step = 0
                    return action

                # Jacobian IK
                jacp = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacp, None, self._env._ee_site_id)

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

                action[6] = -0.5  # keep scoop closed
            else:
                # Fallback: raise and retract
                action[2] = 0.5
                action[3] = -0.6
                action[6] = -0.5
                if self._phase_step > 50:
                    self._phase = 1
                    self._phase_step = 0

        elif self._phase == 1:
            # Phase 1: Open scoop
            action[6] = 1.0
            if self._phase_step > 20:
                self._phase = 2
                self._phase_step = 0

        elif self._phase == 2:
            # Phase 2: Return to stowed
            action[2] = 0.3
            action[3] = -0.3
            action[4] = 0.3
            action[6] = 0.0
            if self._phase_step > 25:
                self._done = True

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
        policy = DepositPolicy(seed=42)
        policy.set_env(env)
        successes = 0
        for ep in range(args.episodes):
            obs, info = env.reset(seed=ep, options={"task": "deposit extracted material"})
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
        print(f"DepositPolicy: {successes}/{args.episodes} = {rate:.1f}% success")
