#!/usr/bin/env python3
"""
Launch SageMaker Training Job for Sports Domain LLM.

This script configures and launches a SageMaker training job.
Run this from your local machine or a SageMaker notebook.

Usage:
    python sagemaker/launch_training.py \
        --train-data s3://your-bucket/data/train \
        --tokenizer-data s3://your-bucket/tokenizer \
        --output-path s3://your-bucket/output \
        --instance-type ml.p4d.24xlarge \
        --model-size medium
"""

import argparse
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput


def parse_args():
    parser = argparse.ArgumentParser(description="Launch SageMaker training job")

    # S3 paths
    parser.add_argument("--train-data", type=str, required=True, help="S3 path to training data")
    parser.add_argument("--tokenizer-data", type=str, required=True, help="S3 path to tokenizer")
    parser.add_argument("--output-path", type=str, required=True, help="S3 path for output")

    # Instance configuration
    parser.add_argument("--instance-type", type=str, default="ml.g5.2xlarge",
                        help="SageMaker instance type")
    parser.add_argument("--instance-count", type=int, default=1,
                        help="Number of instances")
    parser.add_argument("--volume-size", type=int, default=500,
                        help="EBS volume size in GB")

    # Model configuration
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "medium", "large"])
    parser.add_argument("--max-seq-length", type=int, default=2048)

    # Training configuration
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)

    # Job configuration
    parser.add_argument("--job-name", type=str, default=None,
                        help="SageMaker job name (auto-generated if not provided)")
    parser.add_argument("--role", type=str, default=None,
                        help="SageMaker execution role ARN")
    parser.add_argument("--max-run", type=int, default=3600 * 24 * 5,
                        help="Max training time in seconds (default: 5 days)")

    # W&B integration
    parser.add_argument("--wandb-project", type=str, default="sports-llm-sagemaker")
    parser.add_argument("--wandb-api-key", type=str, default=None)

    # Spot instances
    parser.add_argument("--use-spot", action="store_true",
                        help="Use spot instances for cost savings")
    parser.add_argument("--max-wait", type=int, default=3600 * 24 * 7,
                        help="Max wait time for spot instances")

    return parser.parse_args()


def main():
    args = parse_args()

    # Get SageMaker session and role
    sagemaker_session = sagemaker.Session()
    role = args.role or sagemaker.get_execution_role()

    print(f"SageMaker session region: {sagemaker_session.boto_region_name}")
    print(f"Using role: {role}")

    # Hyperparameters
    hyperparameters = {
        "model-size": args.model_size,
        "max-seq-length": args.max_seq_length,
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "gradient-accumulation-steps": args.gradient_accumulation_steps,
        "learning-rate": args.learning_rate,
        "warmup-steps": args.warmup_steps,
        "save-steps": 1000,
        "logging-steps": 10,
        "use-amp": "",  # Flag
    }

    if args.max_steps:
        hyperparameters["max-steps"] = args.max_steps

    if args.wandb_project:
        hyperparameters["wandb-project"] = args.wandb_project

    if args.wandb_api_key:
        hyperparameters["wandb-api-key"] = args.wandb_api_key

    # Distribution configuration for multi-GPU/multi-node
    distribution = None
    if args.instance_count > 1 or "p4d" in args.instance_type or "p5" in args.instance_type:
        distribution = {
            "torch_distributed": {
                "enabled": True
            }
        }

    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="sagemaker",
        role=role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size=args.volume_size,
        framework_version="2.1.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=args.output_path,
        sagemaker_session=sagemaker_session,
        distribution=distribution,
        max_run=args.max_run,
        use_spot_instances=args.use_spot,
        max_wait=args.max_wait if args.use_spot else None,
        checkpoint_s3_uri=f"{args.output_path}/checkpoints" if args.use_spot else None,
        environment={
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        },
        # Include source code
        dependencies=["src", "configs"],
    )

    # Define input channels
    inputs = {
        "train": TrainingInput(
            s3_data=args.train_data,
            content_type="application/jsonlines",
        ),
        "tokenizer": TrainingInput(
            s3_data=args.tokenizer_data,
            content_type="application/x-tar",
        ),
    }

    # Launch training job
    print(f"\nLaunching training job...")
    print(f"Instance type: {args.instance_type} x {args.instance_count}")
    print(f"Model size: {args.model_size}")
    print(f"Training data: {args.train_data}")
    print(f"Output: {args.output_path}")

    if args.job_name:
        estimator.fit(inputs, job_name=args.job_name)
    else:
        estimator.fit(inputs)

    print(f"\nTraining job completed!")
    print(f"Model artifacts: {estimator.model_data}")


if __name__ == "__main__":
    main()
