"""Fine-tuning module for instruction tuning the Sports LLM."""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

from ..models.transformer import SportsLLM
from ..utils.logger import get_logger


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning."""

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0

    # Learning rate (typically lower than pretraining)
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    # Mixed precision
    use_amp: bool = True
    dtype: str = "bfloat16"

    # Checkpointing
    output_dir: str = "./outputs/finetune"
    save_steps: int = 500

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    wandb_project: str = "sports-llm-sft"


class FineTuner:
    """Fine-tuner for instruction tuning."""

    def __init__(
        self,
        model: SportsLLM,
        config: FinetuneConfig,
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

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Calculate total steps
        steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
        self.total_steps = steps_per_epoch * config.num_epochs
        warmup_steps = int(self.total_steps * config.warmup_ratio)

        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=self.total_steps,
            pct_start=config.warmup_ratio,
            anneal_strategy="cos",
        )

        # Setup mixed precision
        self.scaler = GradScaler() if config.use_amp and config.dtype == "float16" else None
        self.amp_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

        # State
        self.global_step = 0

        # Create output dir
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute cross-entropy loss for instruction tuning."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.model(input_ids)
        logits = outputs["logits"]

        # Compute loss (ignore padding tokens with label -100)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        return loss

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate the model."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            if self.config.use_amp:
                with autocast(dtype=self.amp_dtype):
                    loss = self._compute_loss(batch)
            else:
                loss = self._compute_loss(batch)

            total_loss += loss.item()
            num_batches += 1

        self.model.train()
        avg_loss = total_loss / max(1, num_batches)

        return {"eval_loss": avg_loss}

    def save_checkpoint(self, path: str = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")

        Path(path).mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        torch.save(self.model.config, os.path.join(path, "config.pt"))

        self.logger.info(f"Saved checkpoint to {path}")

    def train(self):
        """Run fine-tuning loop."""
        self.logger.info(f"Starting fine-tuning for {self.total_steps} steps")

        # Initialize wandb
        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                config=vars(self.config),
            )

        self.model.train()
        accumulated_loss = 0
        progress_bar = tqdm(total=self.total_steps, desc="Fine-tuning")

        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(self.train_dataloader):
                # Forward pass
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

                accumulated_loss += loss.item() * self.config.gradient_accumulation_steps

                # Optimizer step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    progress_bar.update(1)

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accumulated_loss / self.config.logging_steps
                        lr = self.scheduler.get_last_lr()[0]

                        self.logger.info(
                            f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}"
                        )

                        if self.config.wandb_project:
                            wandb.log(
                                {"train_loss": avg_loss, "learning_rate": lr},
                                step=self.global_step,
                            )

                        accumulated_loss = 0

                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        if eval_metrics:
                            self.logger.info(f"Eval metrics: {eval_metrics}")
                            if self.config.wandb_project:
                                wandb.log(eval_metrics, step=self.global_step)

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

        # Final save
        self.save_checkpoint(os.path.join(self.config.output_dir, "checkpoint-final"))

        progress_bar.close()
        if self.config.wandb_project:
            wandb.finish()

        self.logger.info("Fine-tuning completed!")
