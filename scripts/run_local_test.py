#!/usr/bin/env python3
"""
Local test script to verify the training pipeline works before deploying to SageMaker.

This script:
1. Trains a tokenizer on sample data
2. Creates a small model
3. Runs a few training steps
4. Verifies the model can generate text

Usage:
    python scripts/run_local_test.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.tokenizer.tokenizer import SportsTokenizer
from src.models.config import SportsLLMConfig
from src.models.transformer import SportsLLM
from src.data.dataset import TextDataset, create_dataloader
from src.utils.logger import setup_logger, get_logger


def main():
    setup_logger()
    logger = get_logger()

    # Paths
    data_path = project_root / "data" / "raw" / "sports_sample.jsonl"
    tokenizer_path = project_root / "tokenizer"

    logger.info("="*60)
    logger.info("Sports LLM Local Test")
    logger.info("="*60)

    # Step 1: Train tokenizer
    logger.info("\n[Step 1] Training tokenizer...")

    tokenizer = SportsTokenizer(vocab_size=4000, min_frequency=1)
    tokenizer.train(files=[str(data_path)], show_progress=True)
    tokenizer.save(str(tokenizer_path))

    logger.info(f"Tokenizer trained with vocab size: {tokenizer.vocab_size_actual}")

    # Test tokenizer
    test_text = "Lionel Messi won the World Cup"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    logger.info(f"Test encode/decode: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")

    # Step 2: Create model
    logger.info("\n[Step 2] Creating model...")

    # Use a tiny config for testing
    config = SportsLLMConfig(
        vocab_size=tokenizer.vocab_size_actual,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=512,
    )

    model = SportsLLM(config)
    logger.info(f"Model parameters: {model.num_parameters():,}")
    logger.info(f"Config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")

    # Step 3: Create dataset and dataloader
    logger.info("\n[Step 3] Loading dataset...")

    dataset = TextDataset(
        data_path=str(data_path),
        tokenizer=tokenizer,
        max_length=256,
    )
    logger.info(f"Dataset size: {len(dataset)} samples")

    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    # Step 4: Run a few training steps
    logger.info("\n[Step 4] Running training steps...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for step, batch in enumerate(dataloader):
        if step >= 5:  # Only run 5 steps for testing
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids)
        logits = outputs["logits"]

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logger.info(f"Step {step + 1}: loss = {loss.item():.4f}")

    # Step 5: Test generation
    logger.info("\n[Step 5] Testing generation...")

    model.eval()
    prompt = "The NBA"
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=30,
            temperature=0.8,
            do_sample=True,
        )

    generated_text = tokenizer.decode(generated[0].tolist())
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Generated: '{generated_text}'")

    # Step 6: Save model
    logger.info("\n[Step 6] Saving model...")

    output_path = project_root / "outputs" / "test_model"
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "model.pt")
    torch.save(config, output_path / "config.pt")

    logger.info(f"Model saved to {output_path}")

    logger.info("\n" + "="*60)
    logger.info("Local test completed successfully!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Upload data to S3: python sagemaker/prepare_data.py ...")
    logger.info("2. Launch SageMaker job: python sagemaker/launch_training.py ...")


if __name__ == "__main__":
    main()
