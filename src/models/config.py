"""Model configuration for Sports Domain LLM."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SportsLLMConfig:
    """Configuration for the Sports Domain LLM architecture."""

    # Model dimensions
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048  # FFN hidden size (typically 8/3 * hidden_size for SwiGLU)
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None  # For GQA, None means MHA

    # Sequence
    max_position_embeddings: int = 2048

    # Normalization
    rms_norm_eps: float = 1e-6

    # Dropout
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0

    # Initialization
    initializer_range: float = 0.02

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Attention
    use_flash_attention: bool = True
    rope_theta: float = 10000.0  # RoPE base frequency

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # Validate head dimensions
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"

        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"


# Predefined configurations
CONFIGS = {
    "small": SportsLLMConfig(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
    ),
    "medium": SportsLLMConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=16,
        max_position_embeddings=2048,
    ),
    "large": SportsLLMConfig(
        vocab_size=32000,
        hidden_size=1536,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        max_position_embeddings=4096,
    ),
}


def get_config(name: str) -> SportsLLMConfig:
    """Get a predefined configuration by name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]
