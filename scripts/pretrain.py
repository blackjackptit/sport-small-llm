#!/usr/bin/env python3
"""Pretraining script for Sports Domain LLM."""

import argparse
from pathlib import Path
import yaml
import glob

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.config import SportsLLMConfig, get_config
from src.models.transformer import SportsLLM
from src.tokenizer.tokenizer import SportsTokenizer
from src.data.dataset import TextDataset, StreamingPretrainDataset, create_dataloader
from src.training.pretrain import PreTrainer, PretrainConfig
from src.utils.logger import setup_logger, get_logger


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Pretrain Sports Domain LLM from scratch")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config file",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Model size preset",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer",
        help="Path to trained tokenizer",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pretrain",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logger()
    logger = get_logger()

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = SportsTokenizer.load(args.tokenizer_path)
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size_actual}")

    # Create model config
    logger.info(f"Creating {args.model_size} model")
    model_config = get_config(args.model_size)
    model_config.vocab_size = tokenizer.vocab_size_actual

    # Create model
    model = SportsLLM(model_config)
    logger.info(f"Model parameters: {model.num_parameters():,}")
    logger.info(f"Model config: {model_config}")

    # Load training config
    if Path(args.config).exists():
        train_config_dict = load_yaml_config(args.config)
    else:
        train_config_dict = {}

    # Create training config
    train_config = PretrainConfig(
        output_dir=args.output_dir,
        **{k: v for k, v in train_config_dict.get("training", {}).items()
           if hasattr(PretrainConfig, k)},
    )

    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    data_files = glob.glob(f"{args.data_dir}/**/*.jsonl", recursive=True)
    data_files.extend(glob.glob(f"{args.data_dir}/**/*.txt", recursive=True))

    if not data_files:
        logger.error(f"No data files found in {args.data_dir}")
        logger.info("Please add training data (.jsonl or .txt files)")
        return

    logger.info(f"Found {len(data_files)} data files")

    # Create dataset
    if len(data_files) == 1 and data_files[0].endswith(".jsonl"):
        dataset = TextDataset(
            data_path=data_files[0],
            tokenizer=tokenizer,
            max_length=model_config.max_position_embeddings,
        )
    else:
        dataset = StreamingPretrainDataset(
            data_files=data_files,
            tokenizer=tokenizer,
            max_length=model_config.max_position_embeddings,
        )

    # Create dataloader
    train_dataloader = create_dataloader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
    )

    # Create trainer
    trainer = PreTrainer(
        model=model,
        config=train_config,
        train_dataloader=train_dataloader,
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
