"""Base model configurations and loading utilities."""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def load_base_model(
    model_name: str,
    quantization: str = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
):
    """
    Load a base model for fine-tuning.

    Args:
        model_name: HuggingFace model name or path
        quantization: Quantization config ('4bit', '8bit', or None)
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code

    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    # Quantization config
    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_model_config(model_name: str) -> dict:
    """Get recommended configuration for a given model."""
    configs = {
        "llama": {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_r": 16,
            "lora_alpha": 32,
        },
        "mistral": {
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_r": 16,
            "lora_alpha": 32,
        },
        "default": {
            "target_modules": ["q_proj", "v_proj"],
            "lora_r": 8,
            "lora_alpha": 16,
        },
    }

    for key in configs:
        if key in model_name.lower():
            return configs[key]

    return configs["default"]
