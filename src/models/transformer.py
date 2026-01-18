"""Transformer model for Sports Domain LLM."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .config import SportsLLMConfig
from .layers import RMSNorm, FeedForward
from .attention import MultiHeadAttention, FlashAttention


class TransformerBlock(nn.Module):
    """A single transformer decoder block."""

    def __init__(self, config: SportsLLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Choose attention implementation
        attention_cls = FlashAttention if config.use_flash_attention else MultiHeadAttention

        self.self_attn = attention_cls(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            attention_dropout=config.attention_dropout_prob,
            rope_theta=config.rope_theta,
        )

        self.mlp = FeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
        )

        # Pre-normalization
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class SportsLLM(nn.Module):
    """Sports Domain Large Language Model."""

    def __init__(self, config: SportsLLMConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output projection (weight tying with embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _create_causal_mask(
        self,
        input_shape: Tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        batch_size, seq_len = input_shape

        # Create causal mask
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)

        # Expand for past key values
        if past_key_values_length > 0:
            mask = torch.cat([
                torch.zeros(seq_len, past_key_values_length, dtype=dtype, device=device),
                mask
            ], dim=-1)

        # Expand for batch and heads
        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ):
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Prepare position IDs
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_len + past_key_values_length,
                device=input_ids.device
            ).unsqueeze(0)

        # Create causal mask
        causal_mask = self._create_causal_mask(
            (batch_size, seq_len),
            hidden_states.dtype,
            hidden_states.device,
            past_key_values_length,
        )

        # Process through transformer layers
        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None

            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            if use_cache:
                present_key_values.append(present_key_value)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {
                "logits": logits,
                "past_key_values": present_key_values,
                "hidden_states": hidden_states,
            }
        return logits, present_key_values

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        eos_token_id = eos_token_id or self.config.eos_token_id
        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Forward pass (only process new tokens if using cache)
            if past_key_values is not None:
                curr_input_ids = generated[:, -1:]
            else:
                curr_input_ids = generated

            outputs = self.forward(
                curr_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Stop at EOS
            if (next_token == eos_token_id).all():
                break

        return generated

    @classmethod
    def from_config(cls, config: SportsLLMConfig) -> "SportsLLM":
        """Create model from config."""
        return cls(config)

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
