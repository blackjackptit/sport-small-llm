# Sports LLM Model Architecture

## Overview

The Sports LLM is a decoder-only Transformer model based on the LLaMA architecture, optimized for sports domain text generation.

## Model Configurations

| Config | Parameters | Layers | Heads | Hidden Dim | Intermediate | Context |
|--------|------------|--------|-------|------------|--------------|---------|
| Small  | ~125M      | 12     | 12    | 768        | 2048         | 2048    |
| Medium | ~350M      | 24     | 16    | 1024       | 4096         | 2048    |
| Large  | ~760M      | 24     | 16    | 1536       | 6144         | 4096    |

## Small Model (~125M) - Detailed Specifications

```
vocab_size:              16,000 (or 32,000)
hidden_size:             768
intermediate_size:       2,048
num_hidden_layers:       12
num_attention_heads:     12
num_key_value_heads:     12
head_dim:                64 (768 / 12)
max_position_embeddings: 2,048
rms_norm_eps:            1e-6
rope_theta:              10,000.0
```

## Architecture Diagram

```mermaid
flowchart TB
    subgraph Input["Input Processing"]
        tokens["Input Tokens<br/>(batch_size, seq_len)"]
        embed["Token Embedding<br/>nn.Embedding(16000, 768)"]
        tokens --> embed
    end

    subgraph Transformer["Transformer Stack (x12 layers)"]
        subgraph Block["Transformer Block"]
            direction TB

            subgraph Attention["Multi-Head Self-Attention"]
                ln1["RMSNorm<br/>(768)"]
                qkv["Linear Projections<br/>Q: 768→768<br/>K: 768→768<br/>V: 768→768"]
                rope["RoPE<br/>Rotary Position<br/>Embedding"]
                split["Split into 12 heads<br/>(batch, 12, seq, 64)"]
                attn["Scaled Dot-Product<br/>Attention<br/>+ Causal Mask"]
                concat["Concat Heads<br/>(batch, seq, 768)"]
                o_proj["Output Projection<br/>768→768"]

                ln1 --> qkv
                qkv --> split
                split --> rope
                rope --> attn
                attn --> concat
                concat --> o_proj
            end

            add1["Residual Add"]

            subgraph FFN["Feed-Forward Network (SwiGLU)"]
                ln2["RMSNorm<br/>(768)"]
                gate["Gate Projection<br/>768→2048"]
                up["Up Projection<br/>768→2048"]
                silu["SiLU Activation"]
                mult["Element-wise<br/>Multiply"]
                down["Down Projection<br/>2048→768"]

                ln2 --> gate
                ln2 --> up
                gate --> silu
                silu --> mult
                up --> mult
                mult --> down
            end

            add2["Residual Add"]
        end
    end

    subgraph Output["Output Processing"]
        final_norm["Final RMSNorm<br/>(768)"]
        lm_head["LM Head<br/>768→16000<br/>(weight tied)"]
        logits["Output Logits<br/>(batch, seq, 16000)"]

        final_norm --> lm_head
        lm_head --> logits
    end

    embed --> Block
    o_proj --> add1
    add1 --> FFN
    down --> add2
    add2 -->|"repeat x12"| final_norm

    style Input fill:#e1f5fe
    style Transformer fill:#fff3e0
    style Output fill:#e8f5e9
    style Attention fill:#fce4ec
    style FFN fill:#f3e5f5
```

## Detailed Component Diagrams

### Multi-Head Attention

```mermaid
flowchart LR
    subgraph Input
        X["Hidden States<br/>(B, S, 768)"]
    end

    subgraph Projections
        Q["Q = W_q · X<br/>(B, S, 768)"]
        K["K = W_k · X<br/>(B, S, 768)"]
        V["V = W_v · X<br/>(B, S, 768)"]
    end

    subgraph Reshape
        Q2["Q: (B, 12, S, 64)"]
        K2["K: (B, 12, S, 64)"]
        V2["V: (B, 12, S, 64)"]
    end

    subgraph RoPE["Rotary Position Embedding"]
        cos["cos(mθ)"]
        sin["sin(mθ)"]
        apply["Q' = Q·cos + rotate(Q)·sin<br/>K' = K·cos + rotate(K)·sin"]
    end

    subgraph Attention
        scores["Scores = Q'K'^T / √64"]
        mask["+ Causal Mask"]
        softmax["Softmax"]
        weighted["Attention · V"]
    end

    subgraph Output
        concat["Concat Heads<br/>(B, S, 768)"]
        proj["O_proj<br/>(B, S, 768)"]
    end

    X --> Q & K & V
    Q --> Q2
    K --> K2
    V --> V2
    Q2 & K2 --> apply
    cos & sin --> apply
    apply --> scores
    scores --> mask
    mask --> softmax
    softmax --> weighted
    V2 --> weighted
    weighted --> concat
    concat --> proj
```

### SwiGLU Feed-Forward Network

```mermaid
flowchart LR
    X["Input<br/>(B, S, 768)"]

    subgraph SwiGLU
        gate["gate_proj<br/>768→2048"]
        up["up_proj<br/>768→2048"]
        silu["SiLU(x)"]
        mult["⊙"]
        down["down_proj<br/>2048→768"]
    end

    Y["Output<br/>(B, S, 768)"]

    X --> gate --> silu --> mult
    X --> up --> mult
    mult --> down --> Y
```

### RMSNorm

```mermaid
flowchart LR
    X["Input x"]
    var["variance = mean(x²)"]
    norm["x̂ = x / √(var + ε)"]
    scale["y = γ · x̂"]
    Y["Output y"]

    X --> var --> norm --> scale --> Y
```

## Full Model Flow

```mermaid
flowchart TB
    subgraph Tokenization
        text["Input Text"]
        tokenizer["BPE Tokenizer<br/>(vocab: 16,000)"]
        ids["Token IDs<br/>[1, 234, 567, ...]"]
        text --> tokenizer --> ids
    end

    subgraph Model["SportsLLM"]
        embed["Embedding Layer<br/>16,000 × 768"]

        subgraph Layers["12 Transformer Blocks"]
            block1["Block 1"]
            block2["Block 2"]
            dots["..."]
            block12["Block 12"]
            block1 --> block2 --> dots --> block12
        end

        norm["Final RMSNorm"]
        head["LM Head<br/>(weight tied)"]

        embed --> Layers --> norm --> head
    end

    subgraph Output
        logits["Logits<br/>(B, S, 16000)"]
        sample["Top-k/Top-p<br/>Sampling"]
        next_token["Next Token"]
        logits --> sample --> next_token
    end

    ids --> embed
    head --> logits

    style Tokenization fill:#e3f2fd
    style Model fill:#fff8e1
    style Output fill:#e8f5e9
```

## Parameter Count Breakdown (Small Model)

| Component | Shape | Parameters |
|-----------|-------|------------|
| Token Embedding | 16,000 × 768 | 12,288,000 |
| **Per Transformer Block:** | | |
| - Q projection | 768 × 768 | 589,824 |
| - K projection | 768 × 768 | 589,824 |
| - V projection | 768 × 768 | 589,824 |
| - O projection | 768 × 768 | 589,824 |
| - Gate projection | 768 × 2048 | 1,572,864 |
| - Up projection | 768 × 2048 | 1,572,864 |
| - Down projection | 2048 × 768 | 1,572,864 |
| - Input LayerNorm | 768 | 768 |
| - Post-Attn LayerNorm | 768 | 768 |
| **Block Total** | | 7,079,424 |
| **12 Blocks Total** | | 84,953,088 |
| Final RMSNorm | 768 | 768 |
| LM Head | (tied with embedding) | 0 |
| **Total Parameters** | | **~97M** |

*Note: With vocab_size=32,000, embedding adds ~24.6M params, totaling ~125M*

## Key Architectural Features

### 1. RoPE (Rotary Position Embedding)
- Encodes position information directly into attention computation
- Enables better generalization to longer sequences
- No learned position embeddings needed

### 2. SwiGLU Activation
- Gated Linear Unit with SiLU (Swish) activation
- Formula: `SwiGLU(x) = SiLU(gate(x)) ⊙ up(x)`
- Better performance than standard ReLU/GELU FFN

### 3. RMSNorm (Root Mean Square Normalization)
- Simpler and faster than LayerNorm
- Only scales, no centering: `y = x / RMS(x) * γ`
- Pre-normalization (before attention/FFN)

### 4. Weight Tying
- LM head shares weights with token embedding
- Reduces parameters and improves coherence

### 5. Causal Masking
- Lower triangular attention mask
- Prevents attending to future tokens
- Essential for autoregressive generation

## Training Configuration

```yaml
# Hyperparameters used for training
max_steps: 5000
batch_size: 4
gradient_accumulation_steps: 4
effective_batch_size: 16
learning_rate: 3e-4
warmup_steps: 50
max_seq_length: 512
optimizer: AdamW
weight_decay: 0.1
mixed_precision: fp16
```

## Inference Configuration

```yaml
# Generation parameters
max_new_tokens: 100
temperature: 0.7
top_k: 50
top_p: 0.9
do_sample: true
```
