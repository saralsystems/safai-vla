"""Base class for scripted expert demonstration policies."""

from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    """Abstract base for heuristic sewer robot controllers.

    Each policy maps observations to 7-DOF actions:
        [base_fwd, base_ang, j1, j2, j3, j4, scoop]
    """

    def __init__(self, noise_scale: float = 0.0, seed: int | None = None):
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self._done = False

    @abstractmethod
    def _compute_action(self, obs: dict) -> np.ndarray:
        """Compute raw action from observation. Shape (7,), range [-1, 1]."""
        ...

    def __call__(self, obs: dict) -> np.ndarray:
        """Compute action with optional noise injection."""
        action = self._compute_action(obs)
        if self.noise_scale > 0:
            noise = self.rng.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action.astype(np.float32)

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        self._done = False

    @property
    def done(self) -> bool:
        """Whether the policy considers its sub-task complete."""
        return self._done

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Language prompt string for this sub-task."""
        ...
