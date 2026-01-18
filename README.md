# Sports Domain LLM

A Large Language Model built from scratch, specifically designed for the sports domain to provide accurate, contextual, and expert-level understanding of sports-related content.

## Purpose

This project aims to build a specialized LLM **from the ground up** that excels in sports-related tasks including:

- **Sports Analytics** - Understanding and generating insights from game statistics, player performance metrics, and historical data
- **Sports News & Commentary** - Generating and summarizing sports news, match reports, and expert analysis
- **Rule Interpretation** - Explaining rules and regulations across various sports disciplines
- **Fantasy Sports** - Assisting with player evaluations, draft strategies, and roster optimization
- **Sports Q&A** - Answering questions about teams, players, tournaments, and sporting events

## Why Build From Scratch?

- **Domain-Specific Tokenizer** - Custom vocabulary optimized for sports terminology, team names, player names, and statistics
- **Efficient Architecture** - Model size tailored to sports domain complexity without unnecessary overhead
- **Full Control** - Complete ownership of architecture decisions, training process, and model behavior
- **Learning Experience** - Deep understanding of transformer architecture and LLM training dynamics

## Target Sports Coverage

- Football (Soccer)
- American Football (NFL)
- Basketball (NBA)
- Baseball (MLB)
- Tennis
- Cricket
- Hockey (NHL)
- Golf
- Combat Sports (Boxing, MMA)
- Olympics & Multi-sport Events

## Project Goals

1. **Domain Expertise** - Build a model that demonstrates deep understanding of sports terminology, concepts, and context
2. **Factual Accuracy** - Ensure high accuracy for sports statistics, historical facts, and current information
3. **Multi-sport Knowledge** - Cover a broad range of sports while maintaining depth in major leagues
4. **Real-time Relevance** - Design architecture to incorporate current season data and recent events

## Model Architecture

The model uses a decoder-only Transformer architecture with the following components:

- **Tokenizer**: Custom BPE tokenizer trained on sports corpus
- **Embedding**: Token + Rotary Position Embeddings (RoPE)
- **Attention**: Multi-head attention with causal masking
- **Feed-Forward**: SwiGLU activation function
- **Normalization**: RMSNorm (pre-normalization)

### Model Configurations

| Config | Parameters | Layers | Heads | Dim | Context |
|--------|------------|--------|-------|-----|---------|
| Small  | ~125M      | 12     | 12    | 768 | 2048    |
| Medium | ~350M      | 24     | 16    | 1024| 2048    |
| Large  | ~760M      | 24     | 16    | 1536| 4096    |

## Project Structure

```
build-fresh-llm/
├── configs/
│   ├── model_config.yaml       # Model architecture config
│   └── training_config.yaml    # Training hyperparameters
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Preprocessed data
│   └── external/               # External data sources
├── notebooks/                  # Jupyter notebooks for experiments
├── scripts/
│   ├── train_tokenizer.py      # Train custom tokenizer
│   ├── pretrain.py             # Pretraining script
│   ├── finetune.py             # Instruction fine-tuning
│   └── inference.py            # Inference script
├── src/
│   ├── data/                   # Data loading & preprocessing
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/                 # Model definitions
│   │   ├── transformer.py      # Transformer architecture
│   │   ├── attention.py        # Attention mechanisms
│   │   ├── layers.py           # Model layers (FFN, RMSNorm)
│   │   └── config.py           # Model configurations
│   ├── tokenizer/              # Custom tokenizer
│   │   └── tokenizer.py
│   ├── training/               # Training pipelines
│   │   ├── pretrain.py         # Pretraining loop
│   │   └── finetune.py         # Fine-tuning loop
│   ├── evaluation/             # Evaluation metrics
│   │   └── metrics.py
│   ├── inference/              # Inference utilities
│   │   └── predictor.py
│   └── utils/                  # Common utilities
│       └── logger.py
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
└── README.md
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd build-fresh-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Update `.env` with your credentials (HuggingFace token, W&B API key)

3. Configure model architecture in `configs/model_config.yaml`

4. Configure training in `configs/training_config.yaml`

### Training Pipeline

#### 1. Train Tokenizer
```bash
python scripts/train_tokenizer.py \
    --data_dir data/raw \
    --vocab_size 32000 \
    --output_dir tokenizer/
```

#### 2. Pretraining
```bash
python scripts/pretrain.py \
    --config configs/training_config.yaml \
    --model_config configs/model_config.yaml
```

#### 3. Instruction Fine-tuning (Optional)
```bash
python scripts/finetune.py \
    --checkpoint outputs/pretrain/checkpoint-final \
    --data data/processed/instructions.jsonl
```

### Inference

```bash
# Single prompt
python scripts/inference.py --model_path ./outputs --prompt "Who won the 2022 FIFA World Cup?"

# Interactive mode
python scripts/inference.py --model_path ./outputs --interactive
```

## Training Phases

### Phase 1: Pretraining
- Train on large corpus of sports text (news, Wikipedia, statistics)
- Next-token prediction objective
- Learn language patterns and sports knowledge

### Phase 2: Instruction Fine-tuning
- Train on curated Q&A and instruction-following data
- Supervised fine-tuning (SFT)
- Learn to follow user instructions

### Phase 3: Alignment (Optional)
- RLHF or DPO for response quality
- Improve helpfulness and accuracy

## Tech Stack

- **Framework**: PyTorch
- **Distributed Training**: PyTorch DDP / FSDP
- **Cloud Training**: AWS SageMaker
- **Tokenizer**: SentencePiece / HuggingFace Tokenizers (BPE - Byte Pair Encoding)
- **Experiment Tracking**: Weights & Biases
- **Mixed Precision**: BF16 / FP16 with AMP
- **Serving**: FastAPI, Uvicorn

## AWS SageMaker Training

### Quick Start - Train Small Model (~125M params)

```bash
# One command to train on SageMaker
python sagemaker/launch_small_model.py --s3-bucket your-bucket-name

# With spot instances (65% cheaper)
python sagemaker/launch_small_model.py --s3-bucket your-bucket-name --use-spot

# Custom configuration
python sagemaker/launch_small_model.py \
    --s3-bucket your-bucket-name \
    --instance-type ml.g5.4xlarge \
    --max-steps 5000 \
    --batch-size 8
```

### Prerequisites

1. AWS account with SageMaker access
2. AWS CLI configured (`aws configure`)
3. S3 bucket for data and model artifacts
4. SageMaker execution role with S3 access
5. Python packages: `pip install sagemaker boto3`

### Project Structure for SageMaker

```
sagemaker/
├── train_small.py        # Self-contained small model training
├── train.py              # Full SageMaker entry point
├── launch_small_model.py # Quick launch for small model
├── launch_training.py    # Launch training jobs
├── prepare_data.py       # Upload data to S3
├── config.py             # Instance & cost configurations
└── requirements.txt      # Training dependencies
```

### Step 1: Prepare and Upload Data to S3

```bash
# Upload training data and tokenizer to S3
python sagemaker/prepare_data.py \
    --local-data data/processed \
    --local-tokenizer tokenizer \
    --s3-bucket your-bucket-name \
    --s3-prefix sports-llm
```

### Step 2: Launch Training Job

```bash
# Launch SageMaker training job
python sagemaker/launch_training.py \
    --train-data s3://your-bucket/sports-llm/data/train \
    --tokenizer-data s3://your-bucket/sports-llm/tokenizer \
    --output-path s3://your-bucket/sports-llm/output \
    --instance-type ml.g5.2xlarge \
    --model-size small \
    --epochs 1
```

### Instance Recommendations

| Model Size | Single GPU | Multi-GPU | High Performance |
|------------|------------|-----------|------------------|
| Small (~125M) | ml.g5.2xlarge | ml.g5.12xlarge | ml.p4d.24xlarge |
| Medium (~350M) | ml.g5.4xlarge | ml.g5.48xlarge | ml.p4d.24xlarge |
| Large (~760M) | - | ml.p4d.24xlarge | ml.p5.48xlarge |

### Cost Optimization

- **Spot Instances**: Use `--use-spot` flag for up to 70% cost savings
- **Checkpointing**: Automatic checkpoint saving for spot interruption recovery
- **Right-sizing**: Start with smaller instances, scale up as needed

```bash
# Use spot instances for cost savings
python sagemaker/launch_training.py \
    --train-data s3://your-bucket/sports-llm/data/train \
    --tokenizer-data s3://your-bucket/sports-llm/tokenizer \
    --output-path s3://your-bucket/sports-llm/output \
    --instance-type ml.p4d.24xlarge \
    --model-size medium \
    --use-spot \
    --max-wait 604800  # 7 days max wait
```

### Multi-Node Distributed Training

```bash
# Launch multi-node training on 4 instances
python sagemaker/launch_training.py \
    --train-data s3://your-bucket/sports-llm/data/train \
    --tokenizer-data s3://your-bucket/sports-llm/tokenizer \
    --output-path s3://your-bucket/sports-llm/output \
    --instance-type ml.p4d.24xlarge \
    --instance-count 4 \
    --model-size large
```

### Monitor Training

```bash
# View training logs in CloudWatch
aws logs tail /aws/sagemaker/TrainingJobs --follow

# Or use SageMaker console
# https://console.aws.amazon.com/sagemaker/home#/jobs
```

### Download Trained Model

```bash
# Download model artifacts from S3
aws s3 cp s3://your-bucket/sports-llm/output/model-final ./outputs/model-final --recursive
```

## Data Sources

### Sports News & Articles
| Source | URL | Content Type |
|--------|-----|--------------|
| ESPN | https://www.espn.com | News, scores, analysis |
| BBC Sport | https://www.bbc.com/sport | International sports news |
| The Athletic | https://theathletic.com | In-depth analysis |
| Bleacher Report | https://bleacherreport.com | News, rankings, rumors |
| Sports Illustrated | https://www.si.com | News, features, analysis |
| Yahoo Sports | https://sports.yahoo.com | News, fantasy sports |
| CBS Sports | https://www.cbssports.com | News, scores, odds |
| NBC Sports | https://www.nbcsports.com | News, live coverage |
| Fox Sports | https://www.foxsports.com | News, scores, videos |
| Sky Sports | https://www.skysports.com | UK/European sports |
| The Guardian Sport | https://www.theguardian.com/sport | International coverage |
| Deadspin | https://deadspin.com | Sports culture, news |

### Statistics & Data
| Source | URL | Content Type |
|--------|-----|--------------|
| ESPN Stats | https://www.espn.com/stats | Comprehensive statistics |
| Basketball Reference | https://www.basketball-reference.com | NBA/basketball stats |
| Baseball Reference | https://www.baseball-reference.com | MLB/baseball stats |
| Pro Football Reference | https://www.pro-football-reference.com | NFL stats |
| Hockey Reference | https://www.hockey-reference.com | NHL stats |
| FBref | https://fbref.com | Soccer/football stats |
| Transfermarkt | https://www.transfermarkt.com | Soccer transfers, values |
| Stathead | https://stathead.com | Advanced sports stats |
| Sports Reference | https://www.sports-reference.com | Multi-sport statistics |

### League Official Sites
| Source | URL | Sport |
|--------|-----|-------|
| NFL | https://www.nfl.com | American Football |
| NBA | https://www.nba.com | Basketball |
| MLB | https://www.mlb.com | Baseball |
| NHL | https://www.nhl.com | Hockey |
| FIFA | https://www.fifa.com | Soccer (International) |
| UEFA | https://www.uefa.com | European Soccer |
| Premier League | https://www.premierleague.com | English Soccer |
| La Liga | https://www.laliga.com | Spanish Soccer |
| Serie A | https://www.legaseriea.it | Italian Soccer |
| Bundesliga | https://www.bundesliga.com | German Soccer |
| ATP Tour | https://www.atptour.com | Men's Tennis |
| WTA | https://www.wtatennis.com | Women's Tennis |
| PGA Tour | https://www.pgatour.com | Golf |
| UFC | https://www.ufc.com | MMA |
| WWE | https://www.wwe.com | Wrestling |
| Formula 1 | https://www.formula1.com | Racing |
| Olympics | https://olympics.com | Multi-sport |

### Fantasy & Betting
| Source | URL | Content Type |
|--------|-----|--------------|
| ESPN Fantasy | https://www.espn.com/fantasy | Fantasy sports |
| Yahoo Fantasy | https://sports.yahoo.com/fantasy | Fantasy sports |
| FantasyPros | https://www.fantasypros.com | Fantasy advice, rankings |
| RotoWire | https://www.rotowire.com | Fantasy news, projections |
| DraftKings | https://www.draftkings.com | DFS, betting |
| FanDuel | https://www.fanduel.com | DFS, betting |
| Odds Shark | https://www.oddsshark.com | Betting odds, lines |
| Action Network | https://www.actionnetwork.com | Betting analysis |

### Wikipedia & Reference
| Source | URL | Content Type |
|--------|-----|--------------|
| Wikipedia Sports | https://en.wikipedia.org/wiki/Portal:Sports | Encyclopedia articles |
| Wikimedia Commons | https://commons.wikimedia.org | Sports images, media |

### Social & Community
| Source | URL | Content Type |
|--------|-----|--------------|
| Reddit Sports | https://www.reddit.com/r/sports | Community discussions |
| Reddit NFL | https://www.reddit.com/r/nfl | NFL community |
| Reddit NBA | https://www.reddit.com/r/nba | NBA community |
| Reddit Soccer | https://www.reddit.com/r/soccer | Soccer community |

### APIs for Data Collection
| API | URL | Data Type |
|-----|-----|-----------|
| ESPN API | Internal/Unofficial | Scores, news |
| SportsData.io | https://sportsdata.io | Multi-sport data |
| API-Football | https://www.api-football.com | Soccer data |
| Sportradar | https://sportradar.com | Enterprise sports data |
| The Sports DB | https://www.thesportsdb.com | Free sports data |
| Ball Don't Lie | https://www.balldontlie.io | NBA data |

> **Note**: Always respect robots.txt, terms of service, and rate limits when scraping data. Consider using official APIs where available.

## License

*To be determined*
