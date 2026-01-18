#!/usr/bin/env python3
"""
Launch SageMaker Training Job for Small Sports LLM (~125M parameters).

Usage:
    python sagemaker/launch_small_model.py --s3-bucket your-bucket-name

This will:
1. Upload training data to S3
2. Launch a SageMaker training job
3. Save the model to S3
"""

import argparse
import os
import tarfile
import tempfile
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch


def upload_to_s3(local_path: str, s3_bucket: str, s3_key: str):
    """Upload a file or directory to S3."""
    s3_client = boto3.client("s3")

    local_path = Path(local_path)
    if local_path.is_dir():
        for file in local_path.rglob("*"):
            if file.is_file():
                relative = file.relative_to(local_path)
                key = f"{s3_key}/{relative}"
                print(f"  Uploading {file} -> s3://{s3_bucket}/{key}")
                s3_client.upload_file(str(file), s3_bucket, key)
    else:
        print(f"  Uploading {local_path} -> s3://{s3_bucket}/{s3_key}")
        s3_client.upload_file(str(local_path), s3_bucket, s3_key)


def main():
    parser = argparse.ArgumentParser(description="Launch Small Sports LLM training on SageMaker")

    parser.add_argument("--s3-bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--s3-prefix", type=str, default="sports-llm", help="S3 prefix")
    parser.add_argument("--instance-type", type=str, default="ml.g5.2xlarge",
                        help="SageMaker instance type (default: ml.g5.2xlarge)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--use-spot", action="store_true", help="Use spot instances")
    parser.add_argument("--local-data", type=str, default="data/raw",
                        help="Local path to training data")
    parser.add_argument("--role", type=str, default=None, help="SageMaker execution role ARN")

    args = parser.parse_args()

    print("=" * 60)
    print("Sports LLM - Small Model SageMaker Training")
    print("=" * 60)

    # Get SageMaker session
    sagemaker_session = sagemaker.Session()
    region = sagemaker_session.boto_region_name

    if args.role:
        role = args.role
    else:
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            print("\nERROR: Could not get SageMaker execution role.")
            print("Please provide --role ARN or run from a SageMaker environment.")
            print("\nTo create a role, go to IAM console and create a role with:")
            print("  - AmazonSageMakerFullAccess policy")
            print("  - AmazonS3FullAccess policy (or scoped to your bucket)")
            return

    print(f"\nRegion: {region}")
    print(f"Role: {role}")
    print(f"S3 Bucket: {args.s3_bucket}")

    # Upload training data
    print("\n[1/3] Uploading training data to S3...")
    train_s3_uri = f"s3://{args.s3_bucket}/{args.s3_prefix}/data/train"
    upload_to_s3(args.local_data, args.s3_bucket, f"{args.s3_prefix}/data/train")
    print(f"Training data: {train_s3_uri}")

    # Hyperparameters
    hyperparameters = {
        "max-steps": args.max_steps,
        "batch-size": args.batch_size,
        "gradient-accumulation-steps": 4,
        "learning-rate": 3e-4,
        "warmup-steps": 100,
        "logging-steps": 10,
        "save-steps": 500,
        "max-seq-length": 512,
        "vocab-size": 16000,  # Smaller vocab for sample data
    }

    print("\n[2/3] Creating SageMaker estimator...")
    print(f"Instance type: {args.instance_type}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")

    # Create estimator
    estimator = PyTorch(
        entry_point="train_small.py",
        source_dir="sagemaker",
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        volume_size=50,
        framework_version="2.1.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=f"s3://{args.s3_bucket}/{args.s3_prefix}/output",
        sagemaker_session=sagemaker_session,
        max_run=3600 * 4,  # 4 hours
        use_spot_instances=args.use_spot,
        max_wait=3600 * 8 if args.use_spot else None,
        environment={
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        },
    )

    # Launch training
    print("\n[3/3] Launching training job...")
    print("-" * 60)

    inputs = {
        "train": train_s3_uri,
    }

    estimator.fit(inputs, wait=True)

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel artifacts: {estimator.model_data}")
    print(f"\nTo download the model:")
    print(f"  aws s3 cp {estimator.model_data} ./model.tar.gz")
    print(f"  tar -xzf model.tar.gz")


if __name__ == "__main__":
    main()
