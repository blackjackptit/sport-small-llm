"""
SageMaker Inference Script for Sports LLM

This script handles model loading and inference for the deployed endpoint.
The model architecture must match the training script exactly.
"""

import json
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Model Configuration (must match training)
# ============================================================================

@dataclass
class SmallModelConfig:
    """Configuration for Small Sports LLM (~125M parameters)."""
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    initializer_range: float = 0.02


# ============================================================================
# Model Architecture (copied from train_small.py to match exactly)
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        seq_len = position_ids.max() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, config: SmallModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.rotary_emb = RotaryPositionEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config: SmallModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SportsLLM(nn.Module):
    def __init__(self, config: SmallModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Weight tying

    def _create_causal_mask(self, batch_size: int, seq_len: int, dtype: torch.dtype, device: torch.device):
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        causal_mask = self._create_causal_mask(batch_size, seq_len, hidden_states.dtype, hidden_states.device)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        return {"logits": logits, "loss": loss}

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.9, do_sample: bool = True):
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            logits = outputs["logits"][:, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if EOS token generated
            if next_token.item() == self.config.eos_token_id:
                break

        return generated


# ============================================================================
# Tokenizer
# ============================================================================

class SimpleTokenizer:
    def __init__(self, tokenizer_path: str):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1

    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list) -> str:
        return self.tokenizer.decode(ids)


# ============================================================================
# SageMaker Inference Functions
# ============================================================================

def model_fn(model_dir):
    """Load the model from the model directory."""
    print(f"Loading model from {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Register SmallModelConfig in __main__ so torch.load can find it
    import sys
    sys.modules['__main__'].SmallModelConfig = SmallModelConfig

    # Load config
    config_path = os.path.join(model_dir, "checkpoint-final", "config.pt")
    if os.path.exists(config_path):
        try:
            config = torch.load(config_path, map_location="cpu", weights_only=False)
            print(f"Loaded config type: {type(config)}")
        except Exception as e:
            print(f"Failed to load config: {e}")
            config = SmallModelConfig()
            print("Using default SmallModelConfig")
    else:
        config = SmallModelConfig()
        print("Config file not found, using default SmallModelConfig")

    # Handle config - convert to SmallModelConfig if dict
    if isinstance(config, dict):
        config = SmallModelConfig(
            vocab_size=config.get("vocab_size", 16000),
            hidden_size=config.get("hidden_size", 768),
            intermediate_size=config.get("intermediate_size", 2048),
            num_hidden_layers=config.get("num_hidden_layers", 12),
            num_attention_heads=config.get("num_attention_heads", 12),
            num_key_value_heads=config.get("num_key_value_heads", 12),
            max_position_embeddings=config.get("max_position_embeddings", 2048),
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            rope_theta=config.get("rope_theta", 10000.0),
            pad_token_id=config.get("pad_token_id", 0),
            bos_token_id=config.get("bos_token_id", 1),
            eos_token_id=config.get("eos_token_id", 2),
        )

    print(f"Model config: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, "
          f"num_layers={config.num_hidden_layers}, num_heads={config.num_attention_heads}")

    # Create model
    model = SportsLLM(config)

    # Load weights
    model_path = os.path.join(model_dir, "checkpoint-final", "model.pt")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully")
    else:
        print(f"Warning: Model weights not found at {model_path}")

    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer_path = os.path.join(model_dir, "checkpoint-final", "tokenizer", "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(model_dir, "tokenizer", "tokenizer.json")

    if os.path.exists(tokenizer_path):
        tokenizer = SimpleTokenizer(tokenizer_path)
        print("Tokenizer loaded successfully")
    else:
        print(f"Warning: Tokenizer not found at {tokenizer_path}")
        tokenizer = None

    return {"model": model, "tokenizer": tokenizer, "device": device, "config": config}


def input_fn(request_body, request_content_type):
    """Parse input data."""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_dict):
    """Generate text based on input prompt."""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]

    if tokenizer is None:
        return {"error": "Tokenizer not loaded"}

    # Extract parameters
    prompt = input_data.get("prompt", "")
    max_new_tokens = input_data.get("max_new_tokens", 100)
    temperature = input_data.get("temperature", 0.8)
    top_k = input_data.get("top_k", 50)
    top_p = input_data.get("top_p", 0.9)

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    # Decode
    generated_ids = output_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        }
    }


def output_fn(prediction, accept):
    """Format the prediction output."""
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
