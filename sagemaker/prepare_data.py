#!/usr/bin/env python3
"""
Prepare and Upload Data to S3 for SageMaker Training.

This script helps prepare local data and upload it to S3 for SageMaker training.

Usage:
    python sagemaker/prepare_data.py \
        --local-data data/processed \
        --local-tokenizer tokenizer \
        --s3-bucket your-bucket-name \
        --s3-prefix sports-llm
"""

import argparse
import os
import tarfile
import tempfile
from pathlib import Path

import boto3
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for SageMaker")

    parser.add_argument("--local-data", type=str, required=True,
                        help="Local path to training data directory")
    parser.add_argument("--local-tokenizer", type=str, required=True,
                        help="Local path to tokenizer directory")
    parser.add_argument("--s3-bucket", type=str, required=True,
                        help="S3 bucket name")
    parser.add_argument("--s3-prefix", type=str, default="sports-llm",
                        help="S3 prefix (folder)")
    parser.add_argument("--region", type=str, default=None,
                        help="AWS region")

    return parser.parse_args()


def upload_directory(local_path: str, s3_bucket: str, s3_prefix: str, s3_client):
    """Upload a directory to S3."""
    local_path = Path(local_path)

    files_to_upload = list(local_path.rglob("*"))
    files_to_upload = [f for f in files_to_upload if f.is_file()]

    print(f"Uploading {len(files_to_upload)} files from {local_path}...")

    for file_path in tqdm(files_to_upload, desc="Uploading"):
        relative_path = file_path.relative_to(local_path)
        s3_key = f"{s3_prefix}/{relative_path}"

        s3_client.upload_file(str(file_path), s3_bucket, s3_key)

    return f"s3://{s3_bucket}/{s3_prefix}"


def create_and_upload_tarball(local_path: str, s3_bucket: str, s3_key: str, s3_client):
    """Create a tarball of a directory and upload to S3."""
    local_path = Path(local_path)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    print(f"Creating tarball of {local_path}...")

    with tarfile.open(tmp_path, "w:gz") as tar:
        for item in local_path.iterdir():
            tar.add(item, arcname=item.name)

    print(f"Uploading tarball to s3://{s3_bucket}/{s3_key}...")
    s3_client.upload_file(tmp_path, s3_bucket, s3_key)

    # Cleanup
    os.unlink(tmp_path)

    return f"s3://{s3_bucket}/{s3_key}"


def main():
    args = parse_args()

    # Create S3 client
    s3_client = boto3.client("s3", region_name=args.region)

    # Check if bucket exists
    try:
        s3_client.head_bucket(Bucket=args.s3_bucket)
        print(f"Using existing bucket: {args.s3_bucket}")
    except Exception:
        print(f"Creating bucket: {args.s3_bucket}")
        if args.region and args.region != "us-east-1":
            s3_client.create_bucket(
                Bucket=args.s3_bucket,
                CreateBucketConfiguration={"LocationConstraint": args.region}
            )
        else:
            s3_client.create_bucket(Bucket=args.s3_bucket)

    # Upload training data
    print("\n" + "="*50)
    print("Uploading Training Data")
    print("="*50)

    train_s3_path = upload_directory(
        args.local_data,
        args.s3_bucket,
        f"{args.s3_prefix}/data/train",
        s3_client
    )
    print(f"Training data uploaded to: {train_s3_path}")

    # Upload tokenizer as tarball
    print("\n" + "="*50)
    print("Uploading Tokenizer")
    print("="*50)

    tokenizer_s3_path = create_and_upload_tarball(
        args.local_tokenizer,
        args.s3_bucket,
        f"{args.s3_prefix}/tokenizer/tokenizer.tar.gz",
        s3_client
    )
    print(f"Tokenizer uploaded to: {tokenizer_s3_path}")

    # Print summary
    print("\n" + "="*50)
    print("Upload Complete!")
    print("="*50)
    print(f"\nUse these S3 paths for SageMaker training:")
    print(f"  --train-data {train_s3_path}")
    print(f"  --tokenizer-data s3://{args.s3_bucket}/{args.s3_prefix}/tokenizer")
    print(f"  --output-path s3://{args.s3_bucket}/{args.s3_prefix}/output")

    print(f"\nExample launch command:")
    print(f"""
python sagemaker/launch_training.py \\
    --train-data {train_s3_path} \\
    --tokenizer-data s3://{args.s3_bucket}/{args.s3_prefix}/tokenizer \\
    --output-path s3://{args.s3_bucket}/{args.s3_prefix}/output \\
    --instance-type ml.g5.2xlarge \\
    --model-size small
""")


if __name__ == "__main__":
    main()
