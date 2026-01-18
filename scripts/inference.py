#!/usr/bin/env python3
"""Inference script for Sports Domain LLM."""

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import SportsLLMPredictor


def main():
    parser = argparse.ArgumentParser(description="Run inference with Sports Domain LLM")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    # Initialize predictor
    predictor = SportsLLMPredictor(args.model_path)
    predictor.load_model()

    if args.interactive:
        print("Sports LLM Interactive Mode")
        print("Type 'quit' to exit\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "quit":
                break

            response = predictor.answer_sports_question(user_input)
            print(f"Assistant: {response}\n")

    elif args.prompt:
        response = predictor.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(response)

    else:
        print("Please provide --prompt or use --interactive mode")


if __name__ == "__main__":
    main()
