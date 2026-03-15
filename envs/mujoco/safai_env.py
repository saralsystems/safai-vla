"""MuJoCo sewer environment with Gymnasium API for SafAI VLA Phase 0."""

import logging
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from envs.mujoco.params import EnvConfig

logger = logging.getLogger(__name__)

ASSET_PATH = Path(__file__).parent / "assets" / "sewer_robot.xml"

TASKS = [
    "navigate to blockage",
    "assess and position for extraction",
    "extract sludge at current position",
    "deposit extracted material",
]

NUM_SLUDGE_BODIES = 15
NUM_ARM_JOINTS = 4
ACTION_DIM = 7  # [base_fwd, base_ang, j1, j2, j3, j4, scoop]


class SafaiVLAEnv(gym.Env):
    """MuJoCo sewer maintenance environment.

    Observation space:
        Dict with front_rgb, wrist_rgb, joint_pos, joint_vel,
        ee_pos, ee_quat, base_pos, task.

    Action space:
        Box(-1, 1, (7,)) = [base_fwd, base_ang, j1, j2, j3, j4, scoop]
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = "rgb_array",
    ):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self._step_count = 0
        self._task = TASKS[0]

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(ASSET_PATH))
        self.data = mujoco.MjData(self.model)

        # Renderers for cameras
        self._front_renderer = mujoco.Renderer(
            self.model,
            height=self.config.robot.front_cam_resolution[0],
            width=self.config.robot.front_cam_resolution[1],
        )
        self._wrist_renderer = mujoco.Renderer(
            self.model,
            height=self.config.robot.wrist_cam_resolution[0],
            width=self.config.robot.wrist_cam_resolution[1],
        )

        # Cache body/joint/actuator/sensor ids
        self._sludge_body_ids = []
        self._sludge_jnt_adrs = []
        for i in range(NUM_SLUDGE_BODIES):
            name = f"sludge_{i}"
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._sludge_body_ids.append(bid)
            jnt_name = f"sludge_{i}_joint"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            self._sludge_jnt_adrs.append(self.model.jnt_qposadr[jid])

        self._base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._scoop_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "scoop_site")
        self._deposit_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "deposit_zone"
        )
        self._front_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_cam")
        self._wrist_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

        # Sensor data indices
        # Sensors are ordered as defined in XML: j1-j4 pos, j1-j4 vel, ee_pos, ee_quat, base_pos
        self._joint_pos_sensor_adr = self.model.sensor_adr[0]  # j1_pos
        self._joint_vel_sensor_adr = self.model.sensor_adr[4]  # j1_vel
        self._ee_pos_sensor_adr = self.model.sensor_adr[8]  # ee_pos
        self._ee_quat_sensor_adr = self.model.sensor_adr[9]  # ee_quat
        self._base_pos_sensor_adr = self.model.sensor_adr[10]  # base_pos

        # RNG for domain randomization
        self._np_random = np.random.default_rng()

        # Spaces
        self.observation_space = spaces.Dict(
            {
                "front_rgb": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                "wrist_rgb": spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
                "joint_pos": spaces.Box(
                    -np.inf, np.inf, shape=(NUM_ARM_JOINTS,), dtype=np.float32
                ),
                "joint_vel": spaces.Box(
                    -np.inf, np.inf, shape=(NUM_ARM_JOINTS,), dtype=np.float32
                ),
                "ee_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "ee_quat": spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
                "base_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "task": spaces.Text(max_length=64),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def _get_obs(self) -> dict:
        """Build observation dictionary from current MuJoCo state."""
        # Render cameras
        self._front_renderer.update_scene(self.data, camera=self._front_cam_id)
        front_rgb = self._front_renderer.render()

        self._wrist_renderer.update_scene(self.data, camera=self._wrist_cam_id)
        wrist_rgb = self._wrist_renderer.render()

        # Sensor data
        sd = self.data.sensordata
        joint_pos = sd[self._joint_pos_sensor_adr : self._joint_pos_sensor_adr + 4].copy()
        joint_vel = sd[self._joint_vel_sensor_adr : self._joint_vel_sensor_adr + 4].copy()

        # Compute ee_pos sensor data address from sensor dim
        ee_pos_start = int(np.sum(self.model.sensor_dim[:8]))
        ee_quat_start = ee_pos_start + 3
        base_pos_start = ee_quat_start + 4

        ee_pos = sd[ee_pos_start : ee_pos_start + 3].copy()
        ee_quat = sd[ee_quat_start : ee_quat_start + 4].copy()
        base_pos = sd[base_pos_start : base_pos_start + 3].copy()

        return {
            "front_rgb": front_rgb.copy(),
            "wrist_rgb": wrist_rgb.copy(),
            "joint_pos": joint_pos.astype(np.float32),
            "joint_vel": joint_vel.astype(np.float32),
            "ee_pos": ee_pos.astype(np.float32),
            "ee_quat": ee_quat.astype(np.float32),
            "base_pos": base_pos.astype(np.float32),
            "task": self._task,
        }

    def _get_sludge_positions(self) -> np.ndarray:
        """Get (N, 3) array of active sludge block positions."""
        positions = []
        for i in range(self._active_sludge_count):
            bid = self._sludge_body_ids[i]
            pos = self.data.xpos[bid].copy()
            positions.append(pos)
        if not positions:
            return np.zeros((0, 3))
        return np.array(positions)

    def _compute_reward(self) -> float:
        """Compute task-specific reward signal."""
        sludge_pos = self._get_sludge_positions()
        if len(sludge_pos) == 0:
            return 0.0

        ee_pos = self.data.site_xpos[self._ee_site_id]
        base_pos = self.data.xpos[self._base_body_id]

        if self._task == "navigate to blockage":
            # Negative distance from base to nearest sludge
            dists = np.linalg.norm(sludge_pos[:, :2] - base_pos[:2], axis=1)
            return -float(np.min(dists))

        elif self._task == "assess and position for extraction":
            # Negative distance from EE to nearest sludge surface
            dists = np.linalg.norm(sludge_pos - ee_pos, axis=1)
            return -float(np.min(dists))

        elif self._task == "extract sludge at current position":
            # Sum of sludge z-positions above floor (higher = extracted)
            return float(np.sum(np.maximum(sludge_pos[:, 2] - 0.1, 0.0)))

        elif self._task == "deposit extracted material":
            # Negative distance of sludge to deposit zone
            deposit_pos = self.data.site_xpos[self._deposit_site_id]
            dists = np.linalg.norm(sludge_pos[:, :2] - deposit_pos[:2], axis=1)
            return -float(np.mean(dists))

        return 0.0

    def _check_success(self) -> bool:
        """Check if current task is complete."""
        sludge_pos = self._get_sludge_positions()
        if len(sludge_pos) == 0:
            return False

        ee_pos = self.data.site_xpos[self._ee_site_id]
        base_pos = self.data.xpos[self._base_body_id]

        if self._task == "navigate to blockage":
            dists = np.linalg.norm(sludge_pos[:, :2] - base_pos[:2], axis=1)
            return bool(np.min(dists) < 0.5)

        elif self._task == "assess and position for extraction":
            dists = np.linalg.norm(sludge_pos - ee_pos, axis=1)
            return bool(np.min(dists) < 0.2)

        elif self._task == "extract sludge at current position":
            # Success if scoop is near sludge and below mid-height (engaged)
            scoop_pos = self.data.site_xpos[self._scoop_site_id]
            scoop_dists = np.linalg.norm(sludge_pos - scoop_pos, axis=1)
            return bool(np.min(scoop_dists) < 0.15)

        elif self._task == "deposit extracted material":
            # Success if scoop has moved to deposit zone area (x < 0.2)
            scoop_pos = self.data.site_xpos[self._scoop_site_id]
            deposit_pos = self.data.site_xpos[self._deposit_site_id]
            return bool(np.linalg.norm(scoop_pos[:2] - deposit_pos[:2]) < 0.5)

        return False

    def _randomize_sludge(self) -> None:
        """Randomize sludge block positions and count."""
        cfg = self.config
        if cfg.randomization.sludge_count:
            self._active_sludge_count = self._np_random.integers(
                cfg.sludge.block_count_min, cfg.sludge.block_count_max + 1
            )
        else:
            self._active_sludge_count = 10

        for i in range(NUM_SLUDGE_BODIES):
            adr = self._sludge_jnt_adrs[i]
            if i < self._active_sludge_count:
                if cfg.randomization.sludge_positions:
                    x = self._np_random.uniform(0.8, 2.8)
                    y = self._np_random.uniform(-0.15, 0.15)
                    z = self._np_random.uniform(0.04, 0.08)
                else:
                    x, y, z = 1.5 + i * 0.15, 0.0, 0.06
                self.data.qpos[adr : adr + 3] = [x, y, z]
                self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]  # identity quat
            else:
                # Move inactive blocks far away (underground)
                self.data.qpos[adr : adr + 3] = [0, 0, -10]
                self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
            self.data.qvel[
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"sludge_{i}_joint")
                ] : self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"sludge_{i}_joint")
                ]
                + 6
            ] = 0

    def _randomize_lighting(self) -> None:
        """Randomize ambient light intensity."""
        if self.config.randomization.lighting_intensity:
            lo, hi = self.config.randomization.lighting_intensity_range
            intensity = self._np_random.uniform(lo, hi)
            self.model.light_diffuse[0] = [intensity] * 3

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset environment. Pass task via options={'task': '...'}."""
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Select task
        if options and "task" in options:
            self._task = options["task"]
        else:
            self._task = TASKS[self._np_random.integers(0, len(TASKS))]

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Reset base position
        for jname in ["base_x", "base_y", "base_yaw"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self.data.qpos[self.model.jnt_qposadr[jid]] = 0.0

        # Reset arm to stowed position (slightly raised)
        arm_joints = ["j1", "j2", "j3", "j4"]
        stowed = [0.3, -0.5, 0.5, 0.0]  # safe stowed config
        for jname, val in zip(arm_joints, stowed):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            self.data.qpos[self.model.jnt_qposadr[jid]] = val

        # Randomize sludge
        self._randomize_sludge()
        self._randomize_lighting()

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._initial_sludge_z = self._get_sludge_positions()[:, 2].copy()

        obs = self._get_obs()
        info = {"task": self._task, "active_sludge": self._active_sludge_count}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one environment step."""
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Map 7-DOF action to 8 actuators:
        # action[0] = base_fwd -> actuator 0 (base_fwd velocity)
        # action[1] = base_lat -> actuator 1 (base_lat velocity)
        # action[2:6] = j1-j4  -> actuators 3-6 (arm position)
        # action[6] = scoop    -> actuator 7 (scoop position)
        # actuator 2 (base_yaw) is set to 0

        # Base velocity actuators
        self.data.ctrl[0] = action[0]  # base_fwd
        self.data.ctrl[1] = action[1]  # base_lat
        self.data.ctrl[2] = 0.0  # base_yaw (unused in action space)

        # Arm position actuators: incremental position control
        arm_ranges = [
            (-2.0, 2.0),  # j1
            (-2.5, 2.5),  # j2
            (-2.5, 2.5),  # j3
            (-3.14, 3.14),  # j4
        ]
        for i, (lo, hi) in enumerate(arm_ranges):
            jname = f"j{i + 1}"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            current = self.data.qpos[self.model.jnt_qposadr[jid]]
            delta = action[2 + i] * 0.1  # scale action to reasonable step
            target = np.clip(current + delta, lo, hi)
            self.data.ctrl[3 + i] = target

        # Scoop position actuator (actuator index 7)
        scoop_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "scoop_joint")
        scoop_current = self.data.qpos[self.model.jnt_qposadr[scoop_jid]]
        scoop_delta = action[6] * 0.1
        scoop_target = np.clip(scoop_current + scoop_delta, -0.5, 1.2)
        self.data.ctrl[7] = scoop_target

        # Step simulation (multiple substeps for stability)
        n_substeps = int(1.0 / (self.config.control_frequency_hz * self.model.opt.timestep))
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Compute outputs
        obs = self._get_obs()
        reward = self._compute_reward()
        success = self._check_success()
        terminated = success
        truncated = self._step_count >= self.config.max_episode_steps

        # Count collisions (robot-pipe contacts)
        collision_count = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            if g1_name and g2_name:
                is_pipe = "pipe" in (g1_name or "") or "pipe" in (g2_name or "")
                is_robot = any(
                    k in (g1_name or "") or k in (g2_name or "")
                    for k in ["base", "link", "track", "scoop"]
                )
                if is_pipe and is_robot:
                    collision_count += 1

        info = {
            "task": self._task,
            "success": success,
            "step": self._step_count,
            "collision_count": collision_count,
            "active_sludge": self._active_sludge_count,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render front camera view."""
        if self.render_mode == "rgb_array":
            self._front_renderer.update_scene(self.data, camera=self._front_cam_id)
            return self._front_renderer.render().copy()
        return None

    def close(self) -> None:
        """Clean up renderers."""
        self._front_renderer.close()
        self._wrist_renderer.close()

    def get_sludge_positions(self) -> np.ndarray:
        """Public accessor for sludge positions (used by policies)."""
        return self._get_sludge_positions()

    def get_ee_position(self) -> np.ndarray:
        """Public accessor for end-effector position."""
        return self.data.site_xpos[self._ee_site_id].copy()

    def get_base_position(self) -> np.ndarray:
        """Public accessor for base position."""
        return self.data.xpos[self._base_body_id].copy()

    def get_deposit_position(self) -> np.ndarray:
        """Public accessor for deposit zone position."""
        return self.data.site_xpos[self._deposit_site_id].copy()
