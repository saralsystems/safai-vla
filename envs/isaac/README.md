# Isaac Sim Sewer Environment — Phase 1

This directory will contain the full Isaac Sim sewer environment with:

- OpenUSD procedural pipe generation (parameterized diameter, bends, junctions)
- PhysX 5 PBD particle-based sludge simulation
- Domain randomization via Omniverse Replicator
- Isaac Lab integration for parallelized episode generation
- Cosmos Transfer visual augmentation pipeline

Requires NVIDIA DGX Cloud or local RTX GPU with Isaac Sim installed.

## Status

Pending DGX Cloud credit allocation via NVIDIA Inception program.
MuJoCo proxy environment (`envs/mujoco/`) serves as the Phase 0 stand-in.
Observation space is shared between both environments for model compatibility.
