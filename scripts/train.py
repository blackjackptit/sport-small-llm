#!/usr/bin/env python3
"""Training script for Sports Domain LLM."""

import argparse
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_model import load_base_model
from src.training.trainer import SportsLLMTrainer, SportsLLMTrainingConfig
from src.utils.logger import setup_logger, get_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Sports Domain LLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        help="Path to training data file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for model checkpoints",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logger()
    logger = get_logger()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Override config with CLI arguments
    if args.train_file:
        config["data"]["train_file"] = args.train_file
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

    # Create training config
    training_config = SportsLLMTrainingConfig(
        base_model=config["model"]["name"],
        quantization=config["model"]["quantization"],
        lora_r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        max_seq_length=config["training"]["max_seq_length"],
    )

    # Load base model
    logger.info(f"Loading base model: {training_config.base_model}")
    model, tokenizer = load_base_model(
        training_config.base_model,
        quantization=training_config.quantization,
    )

    # Setup trainer
    trainer = SportsLLMTrainer(training_config)
    trainer.setup_model(model, tokenizer)

    # Load dataset
    # TODO: Implement dataset loading based on config
    logger.info("Dataset loading not implemented - please provide training data")

    # Train
    # trainer.train(train_dataset, eval_dataset)

    # Save
    # trainer.save_model(training_config.output_dir)

    logger.info("Training script setup complete")


if __name__ == "__main__":
    main()
