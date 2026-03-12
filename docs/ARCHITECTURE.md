# Architecture

## Pipeline

```
Synthetic Env → Expert Demos → LeRobot Dataset → SmolVLA LoRA → TensorRT → Jetson
```

## Model

SmolVLA (~1.5B params) is a compact VLA designed for edge deployment. We chose it over OpenVLA (7B) because inference must run on Jetson Orin NX at <100ms per action chunk.

**Vision encoder:** SigLIP-SO (400M) processes two 224×224 RGB views (front camera downscaled + wrist camera native). Dual-view input gives the model both navigation context and manipulation precision.

**Language backbone:** SmolLM2-135M (frozen during fine-tuning). Task conditioning via natural language prompts. Freezing saves compute and prevents catastrophic forgetting of language understanding.

**Action head:** Diffusion-based Action Chunking with Transformers (ACT). Predicts 50-step action trajectories in a single forward pass at 10Hz. Chunked prediction smooths actions and reduces compounding errors vs. single-step prediction.

**Fine-tuning:** LoRA (rank 64) applied to vision encoder Q/V projections and action head layers. Language backbone stays frozen. This keeps the trainable parameter count under 50M — feasible for overnight Mac training and fast iteration.

## Action Space

7-DOF flat vector: `[base_fwd, base_lat, j1, j2, j3, j4, scoop]`

All dimensions normalized to [-1, 1]. The model learns which dimensions matter for each task from the language prompt — navigation uses base dims, manipulation uses arm dims.

## Observation Space

```python
{
    "front_rgb":  (480, 640, 3) uint8,     # front camera
    "wrist_rgb":  (224, 224, 3) uint8,     # wrist camera
    "joint_pos":  (4,) float32,            # arm joint angles
    "joint_vel":  (4,) float32,            # arm joint velocities
    "ee_pos":     (3,) float32,            # end-effector position
    "ee_quat":    (4,) float32,            # end-effector orientation
    "base_pos":   (3,) float32,            # base position in world
    "task":       str,                     # language prompt
}
```

This observation spec is shared between MuJoCo (Phase 0) and Isaac Sim (Phase 1). Models trained on either can be evaluated on both.

## Why Synthetic-Only

No public sewer robotics dataset exists. Real sewer data collection requires physical robots operating in hazardous confined spaces. Synthetic-first with a clear sim-to-real pathway is the only practical approach for bootstrapping this domain.

The sim-to-real gap will be significant — especially for sludge manipulation (rigid blocks vs. particle fluid). This is a known limitation. The moment real episodes become available (even 50), domain adaptation fine-tuning on the synthetic base model closes the gap far faster than training from scratch.
