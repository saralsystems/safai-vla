# Roadmap

## Phase 0 — MuJoCo Proxy (current)

MuJoCo sewer environment with rigid-body sludge blocks. SmolVLA toy checkpoint trained on scripted expert demonstrations. SewerBench evaluation harness. All runnable on a MacBook.

**Goal:** Establish the pipeline, dataset schema, benchmark, and HuggingFace presence. Claim the namespace. Prove the concept is buildable.

## Phase 1 — Isaac Sim Synthetic (next)

Full Isaac Sim environment with PhysX 5 particle-based sludge, procedural pipe generation via OpenUSD, aggressive domain randomization. 10,000+ episodes on DGX Cloud. Production SmolVLA-SewerBot v1 checkpoint.

**Goal:** A model that actually works in simulation. Publishable results on arXiv.

## Phase 2 — Sim-to-Real

Partner with IIT Madras, IIT Bombay, and Genrobotics to collect 50-200 real-robot episodes. Domain adaptation fine-tune on the v1 base. Cosmos Transfer for visual domain bridging. TensorRT deployment on Jetson Orin.

**Goal:** A model that works on a real sewer robot.

## Phase 3 — Municipal Deployment

Smart Cities Mission pilot deployments. Multi-robot fleet coordination. Hindi/regional language operator interface. Training program for municipal staff.

**Goal:** Zero human entry in partner municipalities.
