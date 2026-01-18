#!/usr/bin/env python3
"""Deploy Sports LLM Small Model to SageMaker NOW."""

import boto3
import json
import time
from datetime import datetime

# Configuration
REGION = "eu-central-1"
BUCKET = "sagemaker-eu-central-1-532391786224"
ROLE_ARN = "arn:aws:iam::532391786224:role/AmazonSageMaker-ExecutionRole-20251228T212537"
INSTANCE_TYPE = "ml.g5.2xlarge"

# Job name
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
JOB_NAME = f"sports-llm-small-{timestamp}"

print("=" * 60)
print("Deploying Sports LLM Small Model to SageMaker")
print("=" * 60)
print(f"Region: {REGION}")
print(f"Instance: {INSTANCE_TYPE}")
print(f"Job Name: {JOB_NAME}")

# Create SageMaker client
session = boto3.Session(profile_name="saml", region_name=REGION)
sm_client = session.client("sagemaker")

# Training job configuration
training_params = {
    "TrainingJobName": JOB_NAME,
    "RoleArn": ROLE_ARN,
    "AlgorithmSpecification": {
        "TrainingImage": f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
        "TrainingInputMode": "File",
    },
    "HyperParameters": {
        "max-steps": "500",
        "batch-size": "4",
        "gradient-accumulation-steps": "4",
        "learning-rate": "0.0003",
        "warmup-steps": "50",
        "logging-steps": "10",
        "save-steps": "250",
        "max-seq-length": "512",
        "vocab-size": "16000",
        "sagemaker_program": "train_small.py",
        "sagemaker_submit_directory": f"s3://{BUCKET}/sports-llm/code/sourcedir.tar.gz",
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{BUCKET}/sports-llm/data/train",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }
    ],
    "OutputDataConfig": {
        "S3OutputPath": f"s3://{BUCKET}/sports-llm/output",
    },
    "ResourceConfig": {
        "InstanceType": INSTANCE_TYPE,
        "InstanceCount": 1,
        "VolumeSizeInGB": 50,
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 3600 * 2,  # 2 hours max
    },
}

# Launch training job
print("\nLaunching training job...")
try:
    response = sm_client.create_training_job(**training_params)
    print(f"Training job created: {JOB_NAME}")
    print(f"\nTraining job ARN: {response['TrainingJobArn']}")
except Exception as e:
    print(f"Error: {e}")
    raise

# Monitor job status
print("\nMonitoring training job...")
print("-" * 60)

while True:
    response = sm_client.describe_training_job(TrainingJobName=JOB_NAME)
    status = response["TrainingJobStatus"]
    secondary_status = response.get("SecondaryStatus", "")

    print(f"Status: {status} - {secondary_status}")

    if status in ["Completed", "Failed", "Stopped"]:
        break

    time.sleep(30)

print("-" * 60)

if status == "Completed":
    model_artifacts = response["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"\nTraining completed successfully!")
    print(f"Model artifacts: {model_artifacts}")
    print(f"\nTo download:")
    print(f"  AWS_PROFILE=saml aws s3 cp {model_artifacts} ./model.tar.gz")
elif status == "Failed":
    print(f"\nTraining failed!")
    print(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
else:
    print(f"\nTraining stopped with status: {status}")

print("\n" + "=" * 60)
