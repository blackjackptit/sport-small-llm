#!/usr/bin/env python3
"""
SageMaker Training Entry Point for Sports Domain LLM.

This script is the entry point for SageMaker training jobs.
It handles:
- Loading data from S3 (via SageMaker channels)
- Distributed training setup
- Model checkpointing to S3
- Integration with SageMaker training environment
"""

import os
import sys
import json
import argparse
from pathlib import Path
import glob

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add source to path
sys.path.insert(0, "/opt/ml/code")

from src.models.config import SportsLLMConfig, get_config
from src.models.transformer import SportsLLM
from src.tokenizer.tokenizer import SportsTokenizer
from src.data.dataset import TextDataset, StreamingPretrainDataset, create_dataloader
from src.training.pretrain import PreTrainer, PretrainConfig
from src.utils.logger import setup_logger, get_logger


def setup_distributed():
    """Setup distributed training for SageMaker."""
    # SageMaker sets these environment variables for distributed training
    if "SM_HOSTS" in os.environ:
        hosts = json.loads(os.environ.get("SM_HOSTS", "[]"))
        current_host = os.environ.get("SM_CURRENT_HOST", "")
        num_gpus = int(os.environ.get("SM_NUM_GPUS", 1))

        if len(hosts) > 1 or num_gpus > 1:
            # Initialize distributed training
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)

            return True, local_rank, dist.get_world_size()

    return False, 0, 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Sports LLM on SageMaker")

    # SageMaker specific arguments
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--tokenizer", type=str, default=os.environ.get("SM_CHANNEL_TOKENIZER", "/opt/ml/input/data/tokenizer"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    # Model arguments
    parser.add_argument("--model-size", type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-seq-length", type=int, default=2048)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)

    # Mixed precision
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])

    # W&B
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-api-key", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    setup_logger()
    logger = get_logger()

    # Setup distributed training
    use_distributed, local_rank, world_size = setup_distributed()
    is_main_process = local_rank == 0

    if is_main_process:
        logger.info(f"Starting SageMaker training job")
        logger.info(f"Distributed: {use_distributed}, World size: {world_size}")
        logger.info(f"Train data: {args.train}")
        logger.info(f"Tokenizer: {args.tokenizer}")
        logger.info(f"Model dir: {args.model_dir}")

    # Setup W&B if provided
    if args.wandb_api_key and is_main_process:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        import wandb
        wandb.init(project=args.wandb_project or "sports-llm-sagemaker")

    # Load tokenizer
    tokenizer_path = args.tokenizer
    if os.path.exists(tokenizer_path):
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = SportsTokenizer.load(tokenizer_path)
    else:
        logger.warning(f"Tokenizer not found at {tokenizer_path}, creating new one")
        tokenizer = SportsTokenizer(vocab_size=args.vocab_size)
        # Would need to train tokenizer or use a pre-trained one

    # Create model config
    model_config = get_config(args.model_size)
    model_config.vocab_size = tokenizer.vocab_size_actual or args.vocab_size
    model_config.max_position_embeddings = args.max_seq_length

    if is_main_process:
        logger.info(f"Model config: {model_config}")

    # Create model
    model = SportsLLM(model_config)

    if is_main_process:
        logger.info(f"Model parameters: {model.num_parameters():,}")

    # Move to GPU
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Setup DDP if distributed
    if use_distributed:
        model = DDP(model, device_ids=[local_rank])

    # Load training data
    train_data_path = args.train
    data_files = glob.glob(f"{train_data_path}/**/*.jsonl", recursive=True)
    data_files.extend(glob.glob(f"{train_data_path}/**/*.txt", recursive=True))

    if not data_files:
        raise ValueError(f"No training data found in {train_data_path}")

    if is_main_process:
        logger.info(f"Found {len(data_files)} training files")

    # Create dataset
    if len(data_files) == 1:
        dataset = TextDataset(
            data_path=data_files[0],
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        )
    else:
        dataset = StreamingPretrainDataset(
            data_files=data_files,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
        )

    # Create dataloader with distributed sampler if needed
    if use_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset)
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_dataloader = create_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
        )

    # Create training config
    train_config = PretrainConfig(
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_amp=args.use_amp,
        dtype=args.dtype,
        output_dir=args.model_dir,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        wandb_project=args.wandb_project if is_main_process else None,
        use_ddp=use_distributed,
    )

    # Create trainer
    trainer = PreTrainer(
        model=model,
        config=train_config,
        train_dataloader=train_dataloader,
    )

    # Train
    trainer.train()

    # Save final model (only on main process)
    if is_main_process:
        final_model_path = os.path.join(args.model_dir, "model-final")
        trainer.save_checkpoint(final_model_path)

        # Also save tokenizer
        tokenizer.save(os.path.join(args.model_dir, "tokenizer"))

        logger.info(f"Training complete! Model saved to {args.model_dir}")

    # Cleanup distributed
    if use_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
