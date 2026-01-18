#!/usr/bin/env python3
"""Script to train a custom tokenizer on sports domain corpus."""

import argparse
from pathlib import Path
import glob

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenizer.tokenizer import SportsTokenizer
from src.utils.logger import setup_logger, get_logger


def collect_training_files(data_dir: str, extensions: list = None) -> list:
    """Collect all text files from data directory."""
    if extensions is None:
        extensions = [".txt", ".json", ".jsonl"]

    data_path = Path(data_dir)
    files = []

    for ext in extensions:
        files.extend(glob.glob(str(data_path / f"**/*{ext}"), recursive=True))

    return files


def main():
    parser = argparse.ArgumentParser(description="Train custom tokenizer for Sports LLM")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training text files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokenizer",
        help="Output directory for trained tokenizer",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Target vocabulary size",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum token frequency to include in vocabulary",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase all text during training",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logger()
    logger = get_logger()

    logger.info(f"Starting tokenizer training")
    logger.info(f"Vocab size: {args.vocab_size}")
    logger.info(f"Min frequency: {args.min_frequency}")

    # Collect training files
    training_files = collect_training_files(args.data_dir)
    logger.info(f"Found {len(training_files)} training files")

    if not training_files:
        logger.error(f"No training files found in {args.data_dir}")
        logger.info("Please add text files (.txt, .json, .jsonl) to the data directory")
        return

    # Create and train tokenizer
    tokenizer = SportsTokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        lowercase=args.lowercase,
    )

    logger.info("Training tokenizer...")
    tokenizer.train(files=training_files, show_progress=True)

    # Save tokenizer
    output_path = Path(args.output_dir)
    tokenizer.save(str(output_path))
    logger.info(f"Tokenizer saved to {output_path}")

    # Print stats
    logger.info(f"Final vocabulary size: {tokenizer.vocab_size_actual}")

    # Test tokenizer
    test_texts = [
        "Lionel Messi scored a hat-trick in the Champions League final.",
        "LeBron James has won 4 NBA championships with 3 different teams.",
        "The New York Yankees defeated the Boston Red Sox 5-3.",
    ]

    logger.info("\nTokenizer test:")
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        logger.info(f"Original: {text}")
        logger.info(f"Tokens ({len(tokens)}): {tokens[:20]}...")
        logger.info(f"Decoded: {decoded}")
        logger.info("")


if __name__ == "__main__":
    main()
