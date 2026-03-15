"""SmolVLA LoRA fine-tuning on safai-vla LeRobot dataset.

Loads a LeRobot-format HuggingFace Arrow dataset, applies LoRA to SmolVLA (or a
fallback stub MLP when SmolVLA is unavailable), and trains with MSE loss on
7-DOF action prediction.

Usage:
    python -m training.finetune --dataset data/lerobot/ --output checkpoints/v0/
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset

from training.config import TrainConfig

logger = logging.getLogger(__name__)

ACTION_DIM = 7
STATE_DIM = 6 + 6 + 3 + 4 + 3  # joint_pos + joint_vel + ee_pos + ee_quat + base_pos = 22


def detect_device(preferred: str) -> torch.device:
    """Auto-detect best available device: mps > cuda > cpu."""
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred not in ("mps", "cuda", "cpu"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SafaiVLADataset(Dataset):
    """Wraps a HuggingFace Arrow dataset for action-prediction training.

    Each sample yields (state_tensor, action_tensor) where state_tensor is the
    concatenation of joint_pos, joint_vel, ee_pos, ee_quat, base_pos.
    """

    def __init__(self, hf_dataset) -> None:
        self.data = hf_dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data[idx]
        parts = []
        for key in ("joint_pos", "joint_vel", "ee_pos", "ee_quat", "base_pos"):
            val = row.get(key)
            if val is not None:
                parts.append(np.asarray(val, dtype=np.float32))
        state = np.concatenate(parts) if parts else np.zeros(STATE_DIM, dtype=np.float32)
        action = np.asarray(row["action"], dtype=np.float32)[:ACTION_DIM]
        return torch.from_numpy(state), torch.from_numpy(action)


# ---------------------------------------------------------------------------
# Stub model (fallback when SmolVLA is unavailable)
# ---------------------------------------------------------------------------

# TODO(phase1): Replace StubVLAModel with real SmolVLA once the HuggingFace
# model is stable and loadable on macOS. The stub is a simple MLP that maps
# flattened proprioceptive state to 7-DOF actions. It ignores images and
# language prompts entirely -- it exists only to validate the training pipeline.


class StubVLAModel(nn.Module):
    """Fallback MLP: state -> action. No vision, no language."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        # Named layers so LoRA can find "q_proj" and "v_proj" as targets
        self.q_proj = nn.Linear(state_dim, 256)
        self.act1 = nn.ReLU()
        self.v_proj = nn.Linear(256, 256)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.act3 = nn.ReLU()
        self.head = nn.Linear(128, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass: state vector -> action vector."""
        x = self.act1(self.q_proj(state))
        x = self.act2(self.v_proj(x))
        x = self.act3(self.fc3(x))
        return self.tanh(self.head(x))


def _try_load_smolvla(config: TrainConfig, device: torch.device):
    """Attempt to load SmolVLA from HuggingFace. Returns (model, is_stub)."""
    try:
        from transformers import AutoModelForVision2Seq  # noqa: F401

        logger.info("Attempting to load SmolVLA: %s", config.model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model = model.to(device)
        logger.info("SmolVLA loaded successfully")
        return model, False
    except Exception as exc:
        logger.warning(
            "Could not load SmolVLA (%s). Falling back to stub MLP model. "
            "This is expected in Phase 0 -- the pipeline still validates end-to-end.",
            exc,
        )
        model = StubVLAModel().to(device)
        return model, True


def apply_lora(model: nn.Module, config: TrainConfig, is_stub: bool):
    """Apply LoRA adapters via peft. Skips LoRA for stub model."""
    if is_stub:
        # Stub model trains all params directly — LoRA wrapping is not needed
        # since we're just validating the training pipeline end-to-end.
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Stub model: trainable params: %d || all params: %d || trainable%%: %.1f",
            trainable,
            total,
            100 * trainable / total,
        )
        return model

    target_modules = config.lora_target_modules
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_training(config: TrainConfig) -> None:
    """Main training entry point."""
    device = detect_device(config.device)
    logger.info("Using device: %s", device)
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run `python -m data.collect` and `python -m data.export_lerobot` first."
        )

    logger.info("Loading dataset from %s", dataset_path)
    raw_ds = load_from_disk(str(dataset_path))

    # Train / validation split
    n_total = len(raw_ds)
    n_train = int(n_total * config.train_split)
    raw_ds = raw_ds.shuffle(seed=config.seed)
    train_ds = SafaiVLADataset(raw_ds.select(range(n_train)))
    val_ds = SafaiVLADataset(raw_ds.select(range(n_train, n_total)))
    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
    )

    # --- Model ---
    model, is_stub = _try_load_smolvla(config, device)
    model = apply_lora(model, config, is_stub)

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader) // config.gradient_accumulation_steps)
    total_steps = config.max_steps if config.max_steps > 0 else steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")
    global_step = 0

    logger.info(
        "Starting training: %d epochs, %d total steps, %d warmup",
        config.num_epochs,
        total_steps,
        warmup_steps,
    )

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (states, actions) in enumerate(train_loader):
            states = states.to(device)
            actions = actions.to(device)

            pred_actions = model(states)
            loss = loss_fn(pred_actions, actions) / config.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * config.gradient_accumulation_steps

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.logging_steps == 0:
                    logger.info(
                        "step=%d epoch=%d loss=%.6f lr=%.2e",
                        global_step,
                        epoch,
                        loss.item() * config.gradient_accumulation_steps,
                        scheduler.get_last_lr()[0],
                    )

                # Validation checkpoint
                if global_step % config.eval_steps == 0:
                    val_loss = compute_val_loss(model, val_loader, loss_fn, device)
                    logger.info("step=%d val_loss=%.6f", global_step, val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, config, output_dir / "best", global_step)
                    model.train()

                # Periodic save
                if global_step % config.save_steps == 0:
                    save_checkpoint(model, config, output_dir / f"step-{global_step}", global_step)

                if config.max_steps > 0 and global_step >= config.max_steps:
                    break

        avg_epoch_loss = epoch_loss / max(1, len(train_loader))
        logger.info("Epoch %d complete | avg_loss=%.6f", epoch, avg_epoch_loss)

        if config.max_steps > 0 and global_step >= config.max_steps:
            logger.info("Reached max_steps=%d, stopping.", config.max_steps)
            break

    # Final save
    save_checkpoint(model, config, output_dir / "final", global_step)
    _save_train_summary(config, output_dir, global_step, best_val_loss)
    logger.info("Training complete. Best validation loss: %.6f", best_val_loss)


@torch.no_grad()
def compute_val_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run validation pass and return average MSE loss."""
    model.eval()  # noqa: WPS421
    total_loss = 0.0
    n_batches = 0
    for states, actions in loader:
        states = states.to(device)
        actions = actions.to(device)
        pred = model(states)
        total_loss += loss_fn(pred, actions).item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def save_checkpoint(model: nn.Module, config: TrainConfig, path: Path, step: int) -> None:
    """Save model weights and config."""
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(path))
    else:
        torch.save(model.state_dict(), path / "model.pt")
    with open(path / "train_config.json", "w") as f:
        json.dump(vars(config), f, indent=2)
    logger.info("Checkpoint saved: %s (step %d)", path, step)


def _save_train_summary(
    config: TrainConfig, output_dir: Path, steps: int, best_loss: float
) -> None:
    """Write a JSON summary of the training run."""
    summary = {
        "model_name": config.model_name,
        "total_steps": steps,
        "best_val_loss": best_loss,
        "lora_rank": config.lora_rank,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "seed": config.seed,
    }
    summary_path = output_dir / "train_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary written to %s", summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLA (or stub) with LoRA on safai-vla dataset"
    )
    parser.add_argument("--dataset", type=str, default="data/lerobot/", help="Dataset path")
    parser.add_argument("--output", type=str, default="checkpoints/v0/", help="Output dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--device", type=str, default=None, help="Force device")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = TrainConfig()
    config.dataset_path = args.dataset
    config.output_dir = args.output
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed

    run_training(config)


if __name__ == "__main__":
    main()
