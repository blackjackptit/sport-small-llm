#!/usr/bin/env python3
"""Fine-tuning script for instruction tuning the Sports LLM."""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.config import SportsLLMConfig
from src.models.transformer import SportsLLM
from src.tokenizer.tokenizer import SportsTokenizer
from src.data.dataset import TextDataset, create_dataloader
from src.training.finetune import FineTuner, FinetuneConfig
from src.utils.logger import setup_logger, get_logger


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Sports Domain LLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to instruction tuning data (JSONL)",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Path to evaluation data (JSONL)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/finetune",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logger()
    logger = get_logger()

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = SportsTokenizer.load(args.tokenizer_path)

    # Load model config and weights
    logger.info(f"Loading model from {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)

    model_config = torch.load(checkpoint_path / "config.pt")
    model = SportsLLM(model_config)
    model.load_state_dict(torch.load(checkpoint_path / "model.pt"))

    logger.info(f"Model parameters: {model.num_parameters():,}")

    # Load datasets
    logger.info(f"Loading training data from {args.data}")
    train_dataset = TextDataset(
        data_path=args.data,
        tokenizer=tokenizer,
        max_length=model_config.max_position_embeddings,
    )
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    eval_dataloader = None
    if args.eval_data:
        logger.info(f"Loading eval data from {args.eval_data}")
        eval_dataset = TextDataset(
            data_path=args.eval_data,
            tokenizer=tokenizer,
            max_length=model_config.max_position_embeddings,
        )
        eval_dataloader = create_dataloader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

    # Create fine-tuning config
    finetune_config = FinetuneConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )

    # Create fine-tuner
    finetuner = FineTuner(
        model=model,
        config=finetune_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    # Start fine-tuning
    finetuner.train()


if __name__ == "__main__":
    main()
