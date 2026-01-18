"""Training pipeline for sports domain LLM."""

from dataclasses import dataclass
from typing import Optional

from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


@dataclass
class SportsLLMTrainingConfig:
    """Configuration for training the sports LLM."""

    # Model
    base_model: str = "meta-llama/Llama-2-7b-hf"
    quantization: str = "4bit"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


class SportsLLMTrainer:
    """Trainer class for sports domain LLM."""

    def __init__(self, config: SportsLLMTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self, model, tokenizer):
        """Setup model with LoRA configuration."""
        self.tokenizer = tokenizer

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        # LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()

        return self.model

    def get_training_args(self) -> TrainingArguments:
        """Get training arguments."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            bf16=True,
            optim="paged_adamw_8bit",
            report_to="wandb",
        )

    def train(self, train_dataset, eval_dataset=None):
        """Run training."""
        training_args = self.get_training_args()

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            max_seq_length=self.config.max_seq_length,
        )

        self.trainer.train()

    def save_model(self, output_path: str):
        """Save the trained model."""
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
