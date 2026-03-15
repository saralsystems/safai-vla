"""Training hyperparameters for SmolVLA LoRA fine-tuning."""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Configuration for SmolVLA-SafaiBot fine-tuning.

    Defaults are tuned for MacBook M4 Pro (Phase 0).
    For DGX Cloud (Phase 1), increase batch_size to 32,
    set device="cuda", enable bf16=True.
    """

    # Model
    model_name: str = "HuggingFaceTB/SmolVLA-base"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05

    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    num_epochs: int = 50
    max_steps: int = -1  # -1 = use epochs
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # Evaluation
    eval_steps: int = 250
    eval_episodes: int = 20
    save_steps: int = 500
    save_total_limit: int = 3

    # Hardware
    device: str = "mps"  # auto-detect: mps > cuda > cpu
    fp16: bool = False  # MPS doesn't handle fp16 training well
    bf16: bool = False  # enable on CUDA with Ampere+
    dataloader_num_workers: int = 0  # 0 for Mac compatibility
    gradient_checkpointing: bool = True  # saves memory

    # Output
    output_dir: str = "checkpoints/v0"
    logging_steps: int = 10
    report_to: str = "none"  # "wandb" when ready
    seed: int = 42

    # Dataset
    dataset_path: str = "data/lerobot/"
    train_split: float = 0.9
    action_chunk_size: int = 50
    control_frequency_hz: float = 10.0
