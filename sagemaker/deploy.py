#!/usr/bin/env python3
"""
Deploy Sports LLM to SageMaker Endpoint.

This script handles:
1. Packaging the model with inference script
2. Uploading to S3
3. Creating SageMaker model
4. Creating endpoint configuration
5. Creating or updating endpoint

Usage:
    python sagemaker/deploy.py \
        --model-dir outputs/sports-llm-improved/checkpoint-final \
        --s3-bucket sagemaker-eu-central-1-532391786224 \
        --endpoint-name sports-llm-endpoint \
        --instance-type ml.g5.xlarge
"""

import argparse
import os
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

import boto3


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy Sports LLM to SageMaker")

    parser.add_argument("--model-dir", type=str, required=True,
                        help="Local path to model checkpoint directory")
    parser.add_argument("--s3-bucket", type=str, required=True,
                        help="S3 bucket name")
    parser.add_argument("--s3-prefix", type=str, default="sports-llm",
                        help="S3 prefix (folder)")
    parser.add_argument("--endpoint-name", type=str, default="sports-llm-endpoint",
                        help="SageMaker endpoint name")
    parser.add_argument("--instance-type", type=str, default="ml.g5.xlarge",
                        help="Instance type for inference")
    parser.add_argument("--instance-count", type=int, default=1,
                        help="Number of instances")
    parser.add_argument("--role", type=str, required=True,
                        help="SageMaker execution role ARN")
    parser.add_argument("--region", type=str, default="eu-central-1",
                        help="AWS region")
    parser.add_argument("--model-version", type=str, default=None,
                        help="Model version tag (auto-generated if not provided)")

    return parser.parse_args()


def package_model(model_dir: str, inference_script: str) -> str:
    """Package model and inference script into a tarball."""
    model_dir = Path(model_dir)
    inference_script = Path(inference_script)

    # Create temp file for tarball
    tmp_file = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    print(f"Packaging model from {model_dir}...")

    with tarfile.open(tmp_path, "w:gz") as tar:
        # Add model files
        for item in model_dir.iterdir():
            print(f"  Adding: {item.name}")
            tar.add(item, arcname=item.name)

        # Add inference script as code/inference.py
        print(f"  Adding: code/inference.py")
        tar.add(inference_script, arcname="code/inference.py")

    print(f"Created tarball: {tmp_path}")
    return tmp_path


def upload_to_s3(local_path: str, s3_bucket: str, s3_key: str, s3_client) -> str:
    """Upload file to S3."""
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    print(f"Uploading to {s3_uri}...")
    s3_client.upload_file(local_path, s3_bucket, s3_key)
    print("Upload complete.")
    return s3_uri


def create_sagemaker_model(model_name: str, model_s3_uri: str, role: str,
                           region: str, sagemaker_client) -> str:
    """Create SageMaker model."""
    # Get PyTorch inference container
    container_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"

    print(f"Creating SageMaker model: {model_name}")

    try:
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": container_uri,
                "ModelDataUrl": model_s3_uri,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": model_s3_uri,
                }
            },
            ExecutionRoleArn=role
        )
        print(f"Model created: {model_name}")
    except sagemaker_client.exceptions.ClientError as e:
        if "already exists" in str(e):
            print(f"Model {model_name} already exists, deleting and recreating...")
            sagemaker_client.delete_model(ModelName=model_name)
            time.sleep(2)
            return create_sagemaker_model(model_name, model_s3_uri, role, region, sagemaker_client)
        raise

    return model_name


def create_endpoint_config(config_name: str, model_name: str,
                           instance_type: str, instance_count: int,
                           sagemaker_client) -> str:
    """Create endpoint configuration."""
    print(f"Creating endpoint config: {config_name}")

    try:
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": instance_count,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }]
        )
        print(f"Endpoint config created: {config_name}")
    except sagemaker_client.exceptions.ClientError as e:
        if "already exists" in str(e):
            print(f"Endpoint config {config_name} already exists, deleting and recreating...")
            sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
            time.sleep(2)
            return create_endpoint_config(config_name, model_name, instance_type,
                                         instance_count, sagemaker_client)
        raise

    return config_name


def create_or_update_endpoint(endpoint_name: str, config_name: str,
                              sagemaker_client) -> str:
    """Create or update endpoint."""
    try:
        # Check if endpoint exists
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]

        print(f"Endpoint {endpoint_name} exists (status: {status})")
        print(f"Updating endpoint with new config: {config_name}")

        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )

    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e):
            print(f"Creating new endpoint: {endpoint_name}")
            sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
        else:
            raise

    return endpoint_name


def wait_for_endpoint(endpoint_name: str, sagemaker_client, timeout: int = 900):
    """Wait for endpoint to be in service."""
    print(f"\nWaiting for endpoint to be ready...")

    start_time = time.time()
    while True:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]

        elapsed = int(time.time() - start_time)
        print(f"  [{elapsed}s] Status: {status}")

        if status == "InService":
            print("\nEndpoint is ready!")
            return True
        elif status in ["Failed", "RollingBack"]:
            print(f"\nEndpoint deployment failed: {status}")
            if "FailureReason" in response:
                print(f"Reason: {response['FailureReason']}")
            return False

        if elapsed > timeout:
            print(f"\nTimeout after {timeout}s")
            return False

        time.sleep(30)


def main():
    args = parse_args()

    # Generate version tag
    version = args.model_version or datetime.now().strftime("%Y%m%d-%H%M%S")

    print("=" * 60)
    print("Sports LLM - SageMaker Deployment")
    print("=" * 60)
    print(f"\nModel directory: {args.model_dir}")
    print(f"S3 bucket: {args.s3_bucket}")
    print(f"Endpoint: {args.endpoint_name}")
    print(f"Instance: {args.instance_type} x {args.instance_count}")
    print(f"Version: {version}")

    # Initialize AWS clients
    s3_client = boto3.client("s3", region_name=args.region)
    sagemaker_client = boto3.client("sagemaker", region_name=args.region)

    # Step 1: Package model
    print("\n[1/5] Packaging model...")
    inference_script = Path(__file__).parent / "inference.py"
    tarball_path = package_model(args.model_dir, inference_script)

    # Step 2: Upload to S3
    print("\n[2/5] Uploading to S3...")
    s3_key = f"{args.s3_prefix}/model-{version}.tar.gz"
    model_s3_uri = upload_to_s3(tarball_path, args.s3_bucket, s3_key, s3_client)

    # Cleanup temp file
    os.unlink(tarball_path)

    # Step 3: Create SageMaker model
    print("\n[3/5] Creating SageMaker model...")
    model_name = f"{args.endpoint_name}-model-{version}"
    create_sagemaker_model(model_name, model_s3_uri, args.role,
                          args.region, sagemaker_client)

    # Step 4: Create endpoint config
    print("\n[4/5] Creating endpoint configuration...")
    config_name = f"{args.endpoint_name}-config-{version}"
    create_endpoint_config(config_name, model_name, args.instance_type,
                          args.instance_count, sagemaker_client)

    # Step 5: Create or update endpoint
    print("\n[5/5] Deploying endpoint...")
    create_or_update_endpoint(args.endpoint_name, config_name, sagemaker_client)

    # Wait for endpoint
    success = wait_for_endpoint(args.endpoint_name, sagemaker_client)

    if success:
        print("\n" + "=" * 60)
        print("Deployment Complete!")
        print("=" * 60)
        print(f"\nEndpoint name: {args.endpoint_name}")
        print(f"Region: {args.region}")
        print(f"\nTest with:")
        print(f"""
python -c "
import boto3
import json

client = boto3.client('sagemaker-runtime', region_name='{args.region}')
response = client.invoke_endpoint(
    EndpointName='{args.endpoint_name}',
    ContentType='application/json',
    Body=json.dumps({{'prompt': 'Bradford City', 'max_new_tokens': 100}})
)
result = json.loads(response['Body'].read().decode())
print(result['generated_text'])
"
""")
    else:
        print("\nDeployment failed. Check CloudWatch logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
