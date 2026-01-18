"""Attention mechanisms for the Transformer model."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .layers import RotaryPositionEmbedding, apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with support for Grouped Query Attention (GQA)."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int = 2048,
        attention_dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        self.attention_dropout = nn.Dropout(attention_dropout)

        # Rotary position embeddings
        self.rotary_emb = RotaryPositionEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache for generation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV heads for GQA
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class FlashAttention(MultiHeadAttention):
    """Multi-Head Attention using Flash Attention 2 for efficiency."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Try to use Flash Attention if available
        try:
            from flash_attn import flash_attn_func
            return self._forward_flash(
                hidden_states, attention_mask, position_ids, past_key_value, use_cache
            )
        except ImportError:
            # Fall back to standard attention
            return super().forward(
                hidden_states, attention_mask, position_ids, past_key_value, use_cache
            )

    def _forward_flash(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        from flash_attn import flash_attn_func

        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for flash attention [batch, seq, heads, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        )
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )

        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Need to transpose for apply_rotary_pos_emb
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for GQA
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=2)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=2)

        # Flash attention
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
        )

        # Reshape and project
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value
