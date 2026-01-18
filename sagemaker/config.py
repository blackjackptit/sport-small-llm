"""
SageMaker Configuration Templates for Sports Domain LLM Training.

This module provides pre-configured settings for different training scenarios.
"""

# Instance recommendations based on model size
INSTANCE_RECOMMENDATIONS = {
    "small": {
        "single_gpu": "ml.g5.2xlarge",      # 1x A10G, 24GB
        "multi_gpu": "ml.g5.12xlarge",       # 4x A10G, 96GB total
        "high_performance": "ml.p4d.24xlarge",  # 8x A100, 320GB total
    },
    "medium": {
        "single_gpu": "ml.g5.4xlarge",       # 1x A10G, 24GB (with gradient checkpointing)
        "multi_gpu": "ml.g5.48xlarge",       # 8x A10G, 192GB total
        "high_performance": "ml.p4d.24xlarge",  # 8x A100, 320GB total
    },
    "large": {
        "single_gpu": None,                   # Too large for single GPU
        "multi_gpu": "ml.p4d.24xlarge",      # 8x A100, 320GB total
        "high_performance": "ml.p5.48xlarge",   # 8x H100, 640GB total
    },
}

# Cost estimates (approximate, varies by region)
COST_ESTIMATES_PER_HOUR = {
    "ml.g5.xlarge": 1.41,
    "ml.g5.2xlarge": 1.69,
    "ml.g5.4xlarge": 2.27,
    "ml.g5.8xlarge": 3.42,
    "ml.g5.12xlarge": 7.09,
    "ml.g5.24xlarge": 11.37,
    "ml.g5.48xlarge": 22.74,
    "ml.p4d.24xlarge": 37.69,
    "ml.p5.48xlarge": 98.32,
}

# Training configurations
TRAINING_CONFIGS = {
    "quick_test": {
        "description": "Quick test run to verify setup",
        "max_steps": 100,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "save_steps": 50,
        "logging_steps": 5,
    },
    "development": {
        "description": "Development training with frequent checkpoints",
        "max_steps": 10000,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "save_steps": 500,
        "logging_steps": 10,
    },
    "production_small": {
        "description": "Full training for small model",
        "epochs": 1,
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "save_steps": 2000,
        "logging_steps": 50,
        "warmup_steps": 2000,
    },
    "production_medium": {
        "description": "Full training for medium model",
        "epochs": 1,
        "batch_size": 8,
        "gradient_accumulation_steps": 8,
        "save_steps": 2000,
        "logging_steps": 50,
        "warmup_steps": 3000,
    },
    "production_large": {
        "description": "Full training for large model",
        "epochs": 1,
        "batch_size": 4,
        "gradient_accumulation_steps": 16,
        "save_steps": 2000,
        "logging_steps": 50,
        "warmup_steps": 5000,
    },
}

# S3 path templates
S3_PATH_TEMPLATE = {
    "data": "s3://{bucket}/sports-llm/data/{dataset}",
    "tokenizer": "s3://{bucket}/sports-llm/tokenizer",
    "output": "s3://{bucket}/sports-llm/output/{job_name}",
    "checkpoints": "s3://{bucket}/sports-llm/checkpoints/{job_name}",
}


def get_recommended_instance(model_size: str, budget: str = "multi_gpu") -> str:
    """Get recommended instance type for model size and budget."""
    return INSTANCE_RECOMMENDATIONS.get(model_size, {}).get(budget)


def estimate_training_cost(
    instance_type: str,
    instance_count: int,
    estimated_hours: float,
    use_spot: bool = False,
) -> float:
    """Estimate training cost in USD."""
    hourly_rate = COST_ESTIMATES_PER_HOUR.get(instance_type, 0)
    total_cost = hourly_rate * instance_count * estimated_hours

    if use_spot:
        # Spot instances typically 60-70% cheaper
        total_cost *= 0.35

    return total_cost


def get_training_config(config_name: str) -> dict:
    """Get pre-defined training configuration."""
    return TRAINING_CONFIGS.get(config_name, TRAINING_CONFIGS["development"])


def print_cost_estimate(
    model_size: str,
    instance_type: str,
    instance_count: int,
    estimated_hours: float,
    use_spot: bool = False,
):
    """Print formatted cost estimate."""
    cost = estimate_training_cost(instance_type, instance_count, estimated_hours, use_spot)

    print(f"\n{'='*50}")
    print(f"Cost Estimate for Sports LLM Training")
    print(f"{'='*50}")
    print(f"Model Size: {model_size}")
    print(f"Instance: {instance_type} x {instance_count}")
    print(f"Estimated Duration: {estimated_hours:.1f} hours")
    print(f"Spot Instances: {'Yes' if use_spot else 'No'}")
    print(f"{'='*50}")
    print(f"Estimated Cost: ${cost:.2f} USD")
    print(f"{'='*50}\n")
