#!/usr/bin/env python3
"""
SageMaker Training Script for Small Sports LLM (~125M parameters).

This is a self-contained training script that can run directly on SageMaker
without depending on external source files.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Iterator
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


# ============================================================================
# Model Configuration
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
# Model Layers
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


# ============================================================================
# Full Model
# ============================================================================

class SportsLLM(nn.Module):
    def __init__(self, config: SmallModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

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
                 top_k: int = 50, do_sample: bool = True):
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            logits = outputs["logits"][:, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if (next_token == self.config.eos_token_id).all():
                break

        return generated

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Tokenizer (Simple BPE wrapper)
# ============================================================================

class SimpleTokenizer:
    """Simple tokenizer using HuggingFace tokenizers library."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def train(self, files: List[str]):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        self.tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=1,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
            show_progress=True,
        )
        self.tokenizer.train(files, trainer)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(Path(path) / "tokenizer.json"))
        with open(Path(path) / "config.json", "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "pad_token_id": self.pad_token_id,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "unk_token_id": self.unk_token_id,
            }, f)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        from tokenizers import Tokenizer
        tokenizer = cls()
        tokenizer.tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        with open(Path(path) / "config.json") as f:
            config = json.load(f)
            tokenizer.vocab_size = config["vocab_size"]
            tokenizer.pad_token_id = config["pad_token_id"]
            tokenizer.bos_token_id = config["bos_token_id"]
            tokenizer.eos_token_id = config["eos_token_id"]
            tokenizer.unk_token_id = config["unk_token_id"]
        return tokenizer

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size() if self.tokenizer else self.vocab_size


# ============================================================================
# Dataset
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: SimpleTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        path = Path(data_path)
        if path.is_dir():
            files = list(path.glob("**/*.jsonl")) + list(path.glob("**/*.txt"))
            for f in files:
                self._load_file(f)
        else:
            self._load_file(path)

        print(f"Loaded {len(self.samples)} samples")

    def _load_file(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            if str(path).endswith(".jsonl"):
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if "text" in data and data["text"]:
                            self.samples.append(data["text"])
                    except:
                        continue
            else:
                content = f.read()
                paragraphs = content.split("\n\n")
                self.samples.extend([p.strip() for p in paragraphs if p.strip()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = self.tokenizer.encode(text)

        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        tokens = torch.tensor(tokens, dtype=torch.long)
        input_ids = tokens[:-1]
        labels = tokens[1:].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids, "labels": labels}


# ============================================================================
# Training
# ============================================================================

def train(args):
    print("=" * 60)
    print("Sports LLM Training - Small Model (~125M params)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load or train tokenizer
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else None

    if tokenizer_path and tokenizer_path.exists():
        print(f"\nLoading tokenizer from {tokenizer_path}")
        tokenizer = SimpleTokenizer.load(str(tokenizer_path))
    else:
        print("\nTraining new tokenizer...")
        tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)

        # Find training files
        train_path = Path(args.train)
        if train_path.is_dir():
            files = list(train_path.glob("**/*.jsonl")) + list(train_path.glob("**/*.txt"))
        else:
            files = [train_path]

        tokenizer.train([str(f) for f in files])
        tokenizer.save(args.model_dir + "/tokenizer")

    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    # Create model
    print("\nCreating model...")
    config = SmallModelConfig(vocab_size=tokenizer.get_vocab_size())
    model = SportsLLM(config)
    model.to(device)

    print(f"Model parameters: {model.num_parameters():,}")

    # Create dataset
    print(f"\nLoading data from {args.train}")
    dataset = TextDataset(args.train, tokenizer, max_length=args.max_seq_length)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    total_steps = args.max_steps if args.max_steps else len(dataloader) * args.epochs
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = GradScaler() if args.use_amp and device.type == "cuda" else None
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Training loop
    print(f"\nStarting training for {total_steps} steps...")
    model.train()
    global_step = 0
    accumulated_loss = 0

    progress_bar = tqdm(total=total_steps, desc="Training")

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            if scaler:
                with autocast(dtype=amp_dtype):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs["loss"] / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"] / args.gradient_accumulation_steps
                loss.backward()

            accumulated_loss += loss.item() * args.gradient_accumulation_steps

            # Optimizer step
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = accumulated_loss / args.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    print(f"\nStep {global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                    accumulated_loss = 0

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(model, config, tokenizer, args.model_dir, global_step)

                if args.max_steps and global_step >= args.max_steps:
                    break

        if args.max_steps and global_step >= args.max_steps:
            break

    progress_bar.close()

    # Final save
    save_checkpoint(model, config, tokenizer, args.model_dir, "final")

    # Test generation
    print("\n" + "=" * 60)
    print("Testing generation...")
    model.eval()

    prompts = ["The NBA", "Lionel Messi", "Super Bowl"]
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=30, temperature=0.8)
        text = tokenizer.decode(generated[0].tolist())
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{text}'")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {args.model_dir}")
    print("=" * 60)


def save_checkpoint(model, config, tokenizer, output_dir, step):
    path = Path(output_dir) / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path / "model.pt")
    torch.save(config, path / "config.pt")
    tokenizer.save(str(path / "tokenizer"))

    print(f"Saved checkpoint to {path}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Small Sports LLM")

    # SageMaker paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./outputs"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./data/raw"))
    parser.add_argument("--tokenizer", type=str, default=os.environ.get("SM_CHANNEL_TOKENIZER", None))

    # Model
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-seq-length", type=int, default=512)

    # Training
    parser.add_argument("--epochs", type=int, default=10000)  # Set high default, rely on max-steps to stop
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--use-amp", action="store_true", default=True)

    # Logging
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
