# Contributing to SafAI VLA

We welcome contributions from robotics researchers, ML engineers, municipal technology teams, and anyone who cares about ending manual scavenging in India.

## High-Impact Contributions

**Environment improvements** — better pipe geometry, improved sludge proxy physics, additional sensor modalities, obstacle variation (roots, debris, structural damage).

**Real-world data** — any teleoperation episodes from actual sewer robots, in any format. Even 10 episodes from a real platform are enormously valuable for sim-to-real validation.

**Isaac Sim environment** — USD assets for sewer pipes, particle-based sludge configs, photorealistic material textures from Indian municipal sewers.

**Policy improvements** — better scripted experts, RL-trained policies, or human demonstrations that increase data quality and diversity.

**Evaluation** — additional SafaiBench metrics, real-hardware benchmarking, cross-embodiment transfer testing.

## Getting Started

```bash
git clone https://github.com/saralsystems/safai-vla.git
cd safai-vla
pip install -e ".[dev]"
make test
```

## Code Standards

- Python 3.11+, ruff for lint/format (line-length 99)
- Type hints on all public functions
- Google-style docstrings
- Tests for new environment features
- No GPL dependencies (Apache 2.0 project)

## Pull Requests

1. Fork the repo
2. Create a feature branch from `main`
3. Make your changes, add tests
4. Run `make lint && make test`
5. Open a PR with a clear description of what and why

## Reporting Issues

Open a GitHub issue. Include: what you tried, what happened, what you expected, and your platform (OS, Python version, MuJoCo version).

## Code of Conduct

Be respectful. This project exists to save lives. Act accordingly.
