"""Pretraining loop for Sports Domain LLM."""

import os
import math
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tqdm import tqdm
import wandb

from ..models.transformer import SportsLLM
from ..models.config import SportsLLMConfig
from ..utils.logger import get_logger


@dataclass
class PretrainConfig:
    """Configuration for pretraining."""

    # Training
    num_epochs: int = 1
    max_steps: Optional[int] = None
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Learning rate
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 2000
    lr_scheduler: str = "cosine"  # "cosine" or "linear"

    # Optimization
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Mixed precision
    use_amp: bool = True
    dtype: str = "bfloat16"  # "float16" or "bfloat16"

    # Checkpointing
    output_dir: str = "./outputs/pretrain"
    save_steps: int = 1000
    save_total_limit: int = 3

    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    wandb_project: str = "sports-llm"
    wandb_run_name: Optional[str] = None

    # Distributed
    use_ddp: bool = False


class PreTrainer:
    """Trainer for pretraining the Sports LLM from scratch."""

    def __init__(
        self,
        model: SportsLLM,
        config: PretrainConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.logger = get_logger()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup DDP if needed
        if config.use_ddp and dist.is_initialized():
            self.model = DDP(self.model)
            self.is_main_process = dist.get_rank() == 0
        else:
            self.is_main_process = True

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.total_steps = self._calculate_total_steps()
        self.scheduler = self._create_scheduler()

        # Setup mixed precision
        self.scaler = GradScaler() if config.use_amp and config.dtype == "float16" else None
        self.amp_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "layernorm" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

    def _calculate_total_steps(self) -> int:
        """Calculate total training steps."""
        if self.config.max_steps:
            return self.config.max_steps

        steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_epochs

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        def lr_lambda(step: int) -> float:
            # Warmup
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)

            # Decay
            progress = (step - self.config.warmup_steps) / max(
                1, self.total_steps - self.config.warmup_steps
            )

            if self.config.lr_scheduler == "cosine":
                # Cosine decay to min_lr
                min_lr_ratio = self.config.min_learning_rate / self.config.learning_rate
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                # Linear decay
                min_lr_ratio = self.config.min_learning_rate / self.config.learning_rate
                return max(min_lr_ratio, 1 - progress)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute cross-entropy loss."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.model(input_ids)
        logits = outputs["logits"]

        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        return loss

    def train_step(self, batch: dict) -> float:
        """Execute a single training step."""
        self.model.train()

        # Forward pass with mixed precision
        if self.config.use_amp:
            with autocast(dtype=self.amp_dtype):
                loss = self._compute_loss(batch)
                loss = loss / self.config.gradient_accumulation_steps
        else:
            loss = self._compute_loss(batch)
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        # Optimizer step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate the model."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not self.is_main_process):
            if self.config.use_amp:
                with autocast(dtype=self.amp_dtype):
                    loss = self._compute_loss(batch)
            else:
                loss = self._compute_loss(batch)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_loss, 20))  # Clip for numerical stability

        return {"eval_loss": avg_loss, "perplexity": perplexity}

    def save_checkpoint(self, path: str = None):
        """Save model checkpoint."""
        if not self.is_main_process:
            return

        if path is None:
            path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")

        Path(path).mkdir(parents=True, exist_ok=True)

        # Get model state dict (handle DDP)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        # Save model
        torch.save(model_to_save.state_dict(), os.path.join(path, "model.pt"))

        # Save config
        torch.save(model_to_save.config, os.path.join(path, "config.pt"))

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        if self.scaler:
            training_state["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(training_state, os.path.join(path, "training_state.pt"))

        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        # Load model
        model_state = torch.load(os.path.join(path, "model.pt"), map_location=self.device)
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(model_state)

        # Load training state
        training_state = torch.load(os.path.join(path, "training_state.pt"), map_location=self.device)
        self.global_step = training_state["global_step"]
        self.epoch = training_state["epoch"]
        self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
        self.scheduler.load_state_dict(training_state["scheduler_state_dict"])

        if self.scaler and "scaler_state_dict" in training_state:
            self.scaler.load_state_dict(training_state["scaler_state_dict"])

        self.logger.info(f"Loaded checkpoint from {path}")

    def train(self):
        """Run the full training loop."""
        self.logger.info(f"Starting pretraining for {self.total_steps} steps")
        self.logger.info(f"Model parameters: {self.model.num_parameters():,}")

        # Initialize wandb
        if self.is_main_process and self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )

        accumulated_loss = 0
        progress_bar = tqdm(
            total=self.total_steps,
            desc="Training",
            disable=not self.is_main_process,
        )
        progress_bar.update(self.global_step)

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch

            for step, batch in enumerate(self.train_dataloader):
                # Training step
                loss = self.train_step(batch)
                accumulated_loss += loss

                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    progress_bar.update(1)

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accumulated_loss / self.config.logging_steps
                        lr = self.scheduler.get_last_lr()[0]

                        log_dict = {
                            "train_loss": avg_loss,
                            "learning_rate": lr,
                            "epoch": epoch,
                            "step": self.global_step,
                        }

                        if self.is_main_process:
                            self.logger.info(
                                f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}"
                            )
                            if self.config.wandb_project:
                                wandb.log(log_dict, step=self.global_step)

                        accumulated_loss = 0

                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        if eval_metrics and self.is_main_process:
                            self.logger.info(f"Eval metrics: {eval_metrics}")
                            if self.config.wandb_project:
                                wandb.log(eval_metrics, step=self.global_step)

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

                    # Check if done
                    if self.config.max_steps and self.global_step >= self.config.max_steps:
                        break

            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Final save
        self.save_checkpoint(os.path.join(self.config.output_dir, "checkpoint-final"))

        progress_bar.close()
        if self.is_main_process and self.config.wandb_project:
            wandb.finish()

        self.logger.info("Training completed!")
