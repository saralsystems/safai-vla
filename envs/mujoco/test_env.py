"""Smoke tests for the MuJoCo sewer environment."""

import numpy as np
import pytest

from envs.mujoco.sewer_env import TASKS, SewerVLAEnv


@pytest.fixture
def env():
    """Create a fresh environment for each test."""
    e = SewerVLAEnv()
    yield e
    e.close()


def test_env_creates(env):
    """Environment instantiates without error."""
    assert env is not None
    assert env.model is not None


def test_reset_returns_valid_obs(env):
    """Reset returns observation dict with correct shapes."""
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict)
    assert obs["front_rgb"].shape == (480, 640, 3)
    assert obs["front_rgb"].dtype == np.uint8
    assert obs["wrist_rgb"].shape == (224, 224, 3)
    assert obs["wrist_rgb"].dtype == np.uint8
    assert obs["joint_pos"].shape == (4,)
    assert obs["joint_pos"].dtype == np.float32
    assert obs["joint_vel"].shape == (4,)
    assert obs["ee_pos"].shape == (3,)
    assert obs["ee_quat"].shape == (4,)
    assert obs["base_pos"].shape == (3,)
    assert isinstance(obs["task"], str)
    assert "task" in info


def test_step_returns_correct_types(env):
    """Step accepts random action and returns correct tuple."""
    env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "success" in info
    assert "collision_count" in info


@pytest.mark.parametrize("task", TASKS)
def test_all_tasks_selectable(env, task):
    """Each of the 4 tasks can be selected at reset."""
    obs, info = env.reset(options={"task": task})
    assert obs["task"] == task
    assert info["task"] == task


def test_render_produces_image(env):
    """Render returns an RGB array with correct shape."""
    env.reset(seed=42)
    img = env.render()
    assert img is not None
    assert img.shape == (480, 640, 3)
    assert img.dtype == np.uint8


def test_100_step_rollout(env):
    """100-step rollout completes without error."""
    obs, _ = env.reset(seed=42)
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    assert True


def test_observation_space_contains_obs(env):
    """Observation returned by reset is within the observation space."""
    obs, _ = env.reset(seed=42)
    # Check numeric spaces individually (Text space checked separately)
    for key in [
        "front_rgb",
        "wrist_rgb",
        "joint_pos",
        "joint_vel",
        "ee_pos",
        "ee_quat",
        "base_pos",
    ]:
        assert env.observation_space[key].contains(obs[key]), f"{key} out of bounds"


def test_action_space_shape(env):
    """Action space has correct shape and bounds."""
    assert env.action_space.shape == (7,)
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0


def test_domain_randomization(env):
    """Different seeds produce different sludge configurations."""
    obs1, info1 = env.reset(seed=1)
    sludge1 = env.get_sludge_positions().copy()

    obs2, info2 = env.reset(seed=2)
    sludge2 = env.get_sludge_positions().copy()

    # Positions should differ (probabilistically certain with different seeds)
    if len(sludge1) > 0 and len(sludge2) > 0:
        assert not np.allclose(sludge1[0], sludge2[0])
