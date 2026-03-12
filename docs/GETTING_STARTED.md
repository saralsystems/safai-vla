# Getting Started with sewer-vla

A step-by-step guide for first-time users. No robotics or ML experience required.

## What is this?

sewer-vla is an AI system that learns to control a robot inside sewer pipes. The robot can:
- **Navigate** through pipes to find blockages (sludge)
- **Position** its arm above the sludge
- **Extract** sludge with a scoop
- **Deposit** it in a collection bin

Right now everything runs in simulation (MuJoCo) on your laptop. No real robot needed.

## Prerequisites

- **macOS** (tested on M4 Pro) or Linux
- **Python 3.11** (not 3.12+, not 3.10-)
- **Git**
- ~2 GB free disk space

### Check your Python version

```bash
python3.11 --version
# Should print: Python 3.11.x
```

If you don't have Python 3.11, install it:
```bash
# macOS (with Homebrew)
brew install python@3.11

# Ubuntu/Debian
sudo apt install python3.11 python3.11-venv
```

## Step 1: Clone and Install

```bash
git clone https://github.com/saralsystems/sewer-vla.git
cd sewer-vla

# Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the project
pip install -e ".[dev]"
```

This installs MuJoCo (physics simulator), gymnasium (robotics API), and everything else.

## Step 2: Verify the Environment Works

```bash
python -c "
from envs.mujoco import SewerVLAEnv
env = SewerVLAEnv()
obs, info = env.reset()
print('Environment created!')
print(f'Front camera shape: {obs[\"front_rgb\"].shape}')
print(f'Wrist camera shape: {obs[\"wrist_rgb\"].shape}')
print(f'Action space: {env.action_space.shape}')
env.close()
"
```

You should see:
```
Environment created!
Front camera shape: (480, 640, 3)
Wrist camera shape: (224, 224, 3)
Action space: (7,)
```

## Step 3: Run the Tests

```bash
pytest envs/mujoco/test_env.py -v
```

All 12 tests should pass. If any fail, check:
- Are you using Python 3.11? (`python --version`)
- Is your venv activated? (`which python` should point to `.venv/`)

## Step 4: Watch an Expert Policy

The project includes hand-coded "expert" policies that know how to do each task. Let's see one in action.

```bash
# Test the navigation policy (drives toward sludge)
python -m policies.navigate --test --episodes 10
```

You should see something like:
```
NavigatePolicy: 10/10 = 100.0% success
```

Try the others:
```bash
python -m policies.position --test --episodes 10
python -m policies.extract --test --episodes 10
python -m policies.deposit --test --episodes 10
```

## Step 5: Save a Screenshot

```bash
python -c "
from envs.mujoco import SewerVLAEnv
import imageio

env = SewerVLAEnv()
obs, _ = env.reset(seed=42, options={'task': 'navigate to blockage'})
frame = env.render()
imageio.imwrite('my_first_render.png', frame)
print('Saved my_first_render.png — open it to see the sewer pipe!')
env.close()
"
```

Open `my_first_render.png` to see the simulated pipe interior with the robot and sludge blocks.

## Step 6: Collect Training Data

Expert policies run in the simulator and record what they see and do — this becomes training data for the AI.

```bash
# Collect 20 episodes (quick test, ~30 seconds)
python -m data.collect --task all --episodes 20 --output data/raw_test/

# Verify the data
python -m data.validate --input data/raw_test/
```

For a full dataset (used for training):
```bash
# ~5 minutes on M4 Pro
python -m data.collect --task all --episodes 500 --output data/raw/
```

## Step 7: Export to LeRobot Format

LeRobot is a standard format for robot learning datasets.

```bash
python -m data.export_lerobot --input data/raw_test/ --output data/lerobot_test/
```

## Step 8: Train the AI (Optional)

This trains a small neural network to predict robot actions from sensor data. Requires PyTorch:

```bash
pip install torch peft accelerate datasets

# Quick smoke test (20 training steps, ~10 seconds)
python -m training.finetune \
    --dataset data/lerobot_test/dataset \
    --output checkpoints/test/ \
    --max-steps 20 \
    --batch-size 2 \
    --epochs 1
```

For real training (on the full 500-episode dataset):
```bash
python -m training.finetune \
    --dataset data/lerobot/dataset \
    --output checkpoints/v0/ \
    --epochs 50
```

Note: The model quality will be low in Phase 0 — this is expected. The value is in the working pipeline, not the checkpoint.

## Step 9: Run the Benchmark

SewerBench evaluates policies across all 4 tasks:

```bash
python -m evaluation.sewerbench --episodes 20
```

This generates:
- `evaluation/results/sewerbench_v0.json` — metrics per task
- `evaluation/results/sewerbench_v0.png` — bar chart of success rates

## Understanding the Action Space

The robot has 7 controls, each between -1 and +1:

| Index | What it does | Used for |
|-------|-------------|----------|
| 0 | Drive forward/backward | Navigation |
| 1 | Slide left/right | Navigation |
| 2-5 | Move arm joints 1-4 | Manipulation |
| 6 | Open/close scoop | Extraction |

The AI learns which controls to use based on the task instruction (e.g., "navigate to blockage" → use indices 0-1).

## Understanding the Observations

The robot sees the world through:
- **Front camera** (480x640 RGB) — mounted on the base, looks forward down the pipe
- **Wrist camera** (224x224 RGB) — mounted on the arm, close-up view
- **Joint positions/velocities** — where the arm joints are and how fast they're moving
- **End-effector pose** — position and orientation of the scoop
- **Base position** — where the robot is in the pipe

## Project Structure

```
sewer-vla/
├── envs/mujoco/          # Simulated sewer environment
│   ├── assets/            # MJCF robot model (XML)
│   ├── sewer_env.py       # Gymnasium environment
│   └── test_env.py        # Tests
├── policies/              # Expert controllers
│   ├── navigate.py        # Drive to sludge
│   ├── position.py        # Aim arm at sludge
│   ├── extract.py         # Scoop sludge
│   └── deposit.py         # Drop sludge in bin
├── data/                  # Data pipeline
│   ├── collect.py         # Record expert demos
│   ├── export_lerobot.py  # Convert to HF format
│   └── validate.py        # Quality checks
├── training/              # Model training
│   ├── config.py          # Hyperparameters
│   └── finetune.py        # Training loop
└── evaluation/            # Benchmarking
    ├── sewerbench.py      # Full evaluation suite
    └── metrics.py         # Metric computation
```

## Troubleshooting

### "No module named 'mujoco'"
Your venv isn't activated. Run: `source .venv/bin/activate`

### Tests fail with import errors
Make sure you installed with: `pip install -e ".[dev]"`

### MuJoCo rendering is blank/black
This is headless offscreen rendering — it works but produces dark scenes (sewer pipes are dark!). The brown blocks on the floor are sludge.

### Training runs out of memory
Reduce batch size: `--batch-size 1`

### "Python 3.14 not supported" or similar
Use Python 3.11 specifically. MuJoCo doesn't support 3.12+ yet.

## What's Next?

- Read `docs/ARCHITECTURE.md` for how the AI model works
- Read `docs/ROADMAP.md` for the full project plan
- Phase 1 will use NVIDIA Isaac Sim for realistic physics on DGX Cloud
- Contributions welcome! See `docs/CONTRIBUTING.md`

## Quick Reference

```bash
# Activate environment
source .venv/bin/activate

# Run tests
pytest envs/mujoco/test_env.py -v

# Lint code
ruff check .

# Collect data
python -m data.collect --task all --episodes 500 --output data/raw/

# Export data
python -m data.export_lerobot --input data/raw/ --output data/lerobot/

# Train
python -m training.finetune --dataset data/lerobot/dataset --output checkpoints/v0/

# Evaluate
python -m evaluation.sewerbench --episodes 100
```
