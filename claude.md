# CLAUDE.md — Autonomous Build Guide for sewer-vla

## Project Context

You are building an open-source Vision-Language-Action (VLA) model for autonomous
sewer maintenance robotics in India. Read `app_spec.txt` first — it is the single
source of truth for architecture, scope, and constraints.

This is Phase 0: MuJoCo proxy environment + toy SmolVLA checkpoint, runnable
entirely on a MacBook M4 Pro. Phase 1 (Isaac Sim on DGX Cloud) comes later.

## Critical Rules

1. **Always read `app_spec.txt` before making architectural decisions.**
2. **MuJoCo env must work on macOS without CUDA.** Use MPS for torch, native MuJoCo for sim.
3. **LeRobot dataset format is mandatory.** Do not invent custom formats.
4. **SmolVLA from HuggingFace is the base model.** Do not substitute.
5. **Gymnasium API for all environments.** Standard `reset()` / `step()` interface.
6. **No GPU required for any script to run.** Slow on CPU is acceptable, errors are not.
7. **Apache 2.0 compatible dependencies only.** No GPL.
8. **Keep observation space identical across MuJoCo and future Isaac Sim envs.**

## Build Order

Execute in this exact sequence. Each step must be working and tested before
moving to the next. Do not skip ahead.

### Step 1: Project Skeleton

Set up pyproject.toml, Makefile, .gitignore, pre-commit config, ruff config.

```
# pyproject.toml must include:
[project]
name = "sewer-vla"
version = "0.1.0"
requires-python = ">=3.11"
license = "Apache-2.0"

[project.optional-dependencies]
dev = ["ruff", "pytest", "pre-commit"]
train = ["peft", "accelerate", "wandb"]
```

Makefile targets: `install`, `lint`, `test`, `collect-data`, `train`, `eval`

### Step 2: MJCF Robot Model (`envs/mujoco/assets/sewer_robot.xml`)

Build the MJCF XML for the sewer robot inside a pipe section.

**Pipe:**
- Cylindrical shell using capsule/box geoms, default 900mm ID
- Length ~3m, open at one end (robot entry), closed at other
- Ground plane at pipe bottom

**Robot base:**
- Tracked differential drive (model as box body on slide joints or wheels)
- Width 600mm, fits inside pipe
- 2-DOF base: forward velocity + angular velocity

**Arm:**
- 6 revolute joints, serial chain
- Link lengths proportioned for reaching pipe floor from base-mounted shoulder
- Scoop end-effector: box geom with hinge joint for aperture open/close

**Sludge proxy:**
- 5-15 box geoms at pipe floor, randomized positions
- Free joints so they can be scooped/displaced
- Different sizes (50-200mm)

**Cameras:**
- Front camera: mounted on base, pointing forward, 640x480 implied
- Wrist camera: mounted on last arm link, 224x224 implied

**Validation:** Load in MuJoCo viewer, verify robot fits in pipe, arm reaches
floor, scoop can contact sludge blocks, cameras render.

### Step 3: MuJoCo Environment (`envs/mujoco/sewer_env.py`)

Gymnasium environment wrapping the MJCF model.

```python
class SewerVLAEnv(gymnasium.Env):
    """
    Observation space:
        Dict({
            "front_rgb": Box(0, 255, (480, 640, 3), uint8),
            "wrist_rgb": Box(0, 255, (224, 224, 3), uint8),
            "joint_pos": Box(-inf, inf, (6,), float32),     # arm joints
            "joint_vel": Box(-inf, inf, (6,), float32),     # arm joint vels
            "ee_pos":    Box(-inf, inf, (3,), float32),     # end-effector xyz
            "ee_quat":   Box(-inf, inf, (4,), float32),     # end-effector orientation
            "base_pos":  Box(-inf, inf, (3,), float32),     # base position
            "task":      Text(max_length=64),                # language prompt string
        })

    Action space:
        Box(-1, 1, (7,), float32)
        [0:2] = base velocity (forward, angular)
        [2:8] = arm joint velocity targets (6 joints)
        [8]   = scoop aperture velocity
        # NOTE: base actions [0:1] only used in navigate task
        # Total 9-DOF but we expose 7 by combining base into 2
        # Actually: simplify to 7-DOF:
        #   navigate task: [fwd_vel, ang_vel, 0, 0, 0, 0, 0]
        #   arm tasks: [0, 0, j1, j2, j3, j4, j5, j6, scoop]
        # Wait — keep it at 7 flat: [fwd, ang, j1, j2, j3, j4, scoop]
        # The model learns which dims to use per task from the language prompt.
    """
```

**Important:** Use 7-DOF flat action: `[base_fwd, base_ang, j1, j2, j3, j4, scoop]`.
The arm has 4 joints (not 6) to keep action space manageable for Phase 0.
Phase 1 Isaac Sim env can expand to 6-DOF arm.

**Task selection:** Pass task string at `reset(options={"task": "navigate to blockage"})`.
Default is random task selection.

**Reward (for evaluation, not RL training):**
- navigate: negative distance to nearest sludge block
- position: negative distance from EE to nearest sludge surface
- extract: mass of sludge blocks above pipe floor baseline
- deposit: mass of sludge blocks in deposit zone

**Termination:** max 500 steps, or task-specific success condition met.

**Domain randomization on reset:**
- Pipe diameter: uniform(600, 1200) mm
- Number of sludge blocks: randint(5, 15)
- Sludge block positions: random along pipe floor
- Lighting: vary ambient light intensity

**Tests (`envs/mujoco/test_env.py`):**
- env creates without error
- reset returns valid obs dict with correct shapes
- step accepts random action, returns obs, reward, terminated, truncated, info
- all 4 tasks can be selected
- render produces an image with correct shape
- 100-step rollout completes without error

### Step 4: Scripted Expert Policies (`policies/`)

Each policy is a callable: `action = policy(obs)` → np.array shape (7,).

**`base.py`:** Abstract base class with `__call__(obs) -> action` and `reset()`.

**`navigate.py` — NavigatePolicy:**
- Move base forward toward nearest sludge block
- Simple proportional controller on distance to target
- Stop when within 0.3m of target
- Arm joints held at zero (stowed)

**`position.py` — PositionPolicy:**
- Base stationary
- Inverse-kinematics-like controller to move EE above nearest sludge block
- Use MuJoCo's built-in IK or simple joint-space PD controller toward target
- Success when EE is within 0.05m of sludge surface, oriented downward

**`extract.py` — ExtractPolicy:**
- Lower scoop to pipe floor (joint-space trajectory)
- Close scoop aperture
- Raise arm to lift sludge
- Hard-coded trajectory with noise injection for diversity

**`deposit.py` — DepositPolicy:**
- Move arm to deposit zone (fixed position behind robot)
- Open scoop aperture
- Return to stowed position

**Each policy must:**
- Work deterministically (for validation)
- Accept a `noise_scale` param for diverse demonstrations
- Return `done=True` in info dict when sub-task is complete
- Achieve >90% success over 100 episodes in the MuJoCo env

**Test:** Run each policy for 50 episodes, print success rate.

### Step 5: Data Collection (`data/collect.py`)

Script that runs expert policies in the MuJoCo env and records episodes.

```bash
python -m data.collect --task all --episodes 500 --output data/raw/
```

**Per episode, record at each timestep:**
- observation dict (all fields)
- action (7-DOF)
- reward
- timestamp

**Storage:** Save as HDF5 or pickle per episode. One file per episode.
Include metadata: task name, success (bool), episode length, env params (pipe diameter etc).

**Noise injection:** 70% of episodes use noisy expert (noise_scale=0.1), 30% clean.

### Step 6: LeRobot Export (`data/export_lerobot.py`)

Convert raw episodes to LeRobot HuggingFace dataset format.

```bash
python -m data.export_lerobot --input data/raw/ --output data/lerobot/
```

**Key:** Study the LeRobot dataset format carefully before implementing.
The dataset must be loadable by `lerobot.common.datasets.lerobot_dataset.LeRobotDataset`.

Include in the dataset card:
- Embodiment: "sewer-vla-mujoco-proxy"
- Action space: 7-DOF description
- Observation keys and shapes
- Task language prompts
- Known limitations: rigid-body proxy, not particle fluid
- License: Apache 2.0

### Step 7: Push to HuggingFace (`data/push_to_hub.py`)

```bash
python -m data.push_to_hub --dataset data/lerobot/ --repo saral-systems/sewer-vla-mujoco-v0
```

This is a thin wrapper. It should also generate and upload:
- Dataset card (README.md for HF)
- Episode statistics (mean/std per observation key, per action dim)
- Sample episode GIFs (3-5 episodes rendered as short GIFs)

**NOTE:** If HuggingFace auth is not available, save the dataset locally and
print instructions for manual upload. Do not fail.

### Step 8: SmolVLA Fine-Tuning (`training/finetune.py`)

```bash
python -m training.finetune --dataset data/lerobot/ --output checkpoints/v0/
```

**Config (`training/config.py`):**
```python
@dataclass
class TrainConfig:
    model_name: str = "HuggingFaceTB/SmolVLA-base"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 2e-4
    batch_size: int = 4          # small for Mac
    num_epochs: int = 50
    max_steps: int = -1          # -1 = use epochs
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    save_steps: int = 500
    eval_steps: int = 250
    output_dir: str = "checkpoints/v0"
    device: str = "mps"          # auto-detect: mps > cuda > cpu
    fp16: bool = False           # MPS doesn't support fp16 training well
    bf16: bool = False
    dataloader_num_workers: int = 0  # 0 for Mac compatibility
    seed: int = 42
```

**Training loop:**
1. Load SmolVLA base model
2. Apply LoRA adapters via peft
3. Load LeRobot dataset
4. Standard SFT loop: obs + task_prompt → action prediction, MSE loss on actions
5. Evaluate every eval_steps on held-out episodes
6. Save best checkpoint by eval loss
7. Save LoRA adapter weights separately (small, easy to distribute)

**Important:** If SmolVLA is not yet available or the API has changed, check
HuggingFace for the latest SmolVLA release and adapt. If the model is too large
for Mac memory, reduce batch_size to 1 and enable gradient checkpointing.
If SmolVLA truly cannot load, fall back to a smaller VLA or OpenVLA with a clear
TODO comment marking the substitution.

### Step 9: SewerBench Evaluation (`evaluation/sewerbench.py`)

```bash
python -m evaluation.sewerbench --checkpoint checkpoints/v0/ --episodes 100
```

**Metrics per sub-task:**
- Success rate (%)
- Average completion time (steps)
- Average collision count (robot-pipe contacts)
- For extract: average sludge mass displaced

**Output:**
- JSON results file: `evaluation/results/sewerbench_v0.json`
- Summary plot (bar chart of success rates): `evaluation/results/sewerbench_v0.png`
- 5 sample episode GIFs per sub-task: `evaluation/results/episodes/`

**Baseline comparison:**
- Random policy (expected ~0% success)
- Scripted expert (expected >90% success)
- Trained SmolVLA-SewerBot-v0 (somewhere in between)

### Step 10: Documentation

**README.md:**
- Banner image/SVG
- One-line description
- Architecture mermaid diagram (from assets/architecture.mermaid)
- Quickstart (install, run env, collect data, train, eval)
- Model card summary
- Roadmap (Phase 0 → Phase 1)
- Citation placeholder
- License badge
- Contributing link

**docs/ARCHITECTURE.md:** Detailed architecture, model choices, action space rationale.
**docs/EMBODIMENT.md:** Reference robot spec with dimensions, joint limits, sensor specs.
**docs/SEWER_PARAMS.md:** Indian sewer geometry reference data from CPHEEO/NEERI.
**docs/ROADMAP.md:** Phase 0 → Phase 1 (Isaac Sim) → Phase 2 (sim-to-real) → Phase 3 (fleet).
**docs/CONTRIBUTING.md:** How to contribute — env improvements, real-world data, policy improvements.

### Step 11: CI (`/.github/workflows/ci.yml`)

GitHub Actions workflow:
- Trigger: push to main, PRs
- Python 3.11
- Install deps: `pip install -e ".[dev]"`
- Lint: `ruff check .`
- Test: `pytest envs/mujoco/test_env.py -v`
- Env smoke test: create env, run 10 random steps, verify no crash

## Code Style

- ruff for lint + format (line-length 99)
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- No print statements in library code — use logging
- Scripts (collect.py, finetune.py, etc.) use argparse or typer for CLI
- Keep files under 300 lines. Split if longer.

## Common Pitfalls

- MuJoCo rendering on headless Mac: use `mujoco.Renderer` not `mujoco.viewer`
  for offscreen rendering. The viewer requires a display.
- LeRobot format changes between versions. Pin the version in pyproject.toml
  and check their docs before implementing export.
- SmolVLA may require specific transformers version. Check HF model card.
- MPS backend has quirks: no fp16, some ops fall back to CPU silently.
  Test with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` if OOM.
- HDF5 files: use h5py, not pandas. Keep episode files under 100MB each.
- Camera rendering in MuJoCo: use `mujoco.Renderer(model, height, width)`
  then `renderer.update_scene(data, camera=cam_name)` then `renderer.render()`.

## Testing Commands

```bash
# After Step 3:
pytest envs/mujoco/test_env.py -v

# After Step 4:
python -m policies.navigate --test --episodes 50
python -m policies.position --test --episodes 50
python -m policies.extract --test --episodes 50
python -m policies.deposit --test --episodes 50

# After Step 5:
python -m data.collect --task all --episodes 10 --output data/raw_test/
ls data/raw_test/  # should have 10 files

# After Step 6:
python -m data.export_lerobot --input data/raw_test/ --output data/lerobot_test/

# After Step 9:
python -m evaluation.sewerbench --checkpoint checkpoints/v0/ --episodes 20
cat evaluation/results/sewerbench_v0.json
```

## What Success Looks Like

Phase 0 is successful when:
1. `make test` passes — env creates, steps, renders without error
2. All 4 expert policies achieve >90% success in the MuJoCo env
3. 500+ episodes are collected and exported in LeRobot format
4. SmolVLA LoRA fine-tuning completes without OOM on M4 Pro (even if results are bad)
5. SewerBench produces a JSON report with per-task metrics
6. README is clear enough that a stranger can clone, install, and run eval in <10 minutes
7. HuggingFace dataset and model repos exist (even as local stubs if auth unavailable)

The model WILL be bad. That is expected and acceptable. The value is in the
pipeline, the dataset format, the benchmark, and the repo structure — not the
checkpoint quality. Phase 1 on Isaac Sim with real physics fixes the quality.
