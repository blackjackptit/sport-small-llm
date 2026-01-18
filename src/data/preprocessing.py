"""Data preprocessing utilities for pretraining."""

import json
import re
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Any
from tqdm import tqdm
import torch


def clean_text(text: str) -> str:
    """Clean and normalize text data."""
    if not text:
        return ""

    # Basic cleaning
    text = text.strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def clean_sports_text(text: str) -> str:
    """Clean sports-specific text patterns."""
    text = clean_text(text)

    # Normalize score formats (e.g., "3 - 2" -> "3-2")
    text = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1-\2", text)

    return text


def process_jsonl_file(
    input_path: str,
    output_path: str,
    text_field: str = "text",
    min_length: int = 50,
) -> int:
    """
    Process a JSONL file and write cleaned data.

    Returns number of processed documents.
    """
    count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Processing"):
            try:
                data = json.loads(line.strip())
                text = data.get(text_field, "")

                # Clean text
                text = clean_sports_text(text)

                # Filter by length
                if len(text) >= min_length:
                    output_data = {"text": text}
                    fout.write(json.dumps(output_data) + "\n")
                    count += 1

            except json.JSONDecodeError:
                continue

    return count


def tokenize_corpus(
    input_files: List[str],
    output_path: str,
    tokenizer,
    max_length: Optional[int] = None,
) -> int:
    """
    Tokenize an entire corpus and save as a single tensor.

    Returns total number of tokens.
    """
    all_tokens = []

    for file_path in tqdm(input_files, desc="Tokenizing files"):
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".jsonl"):
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        text = data.get("text", "")
                        if text:
                            tokens = tokenizer.encode(text, add_special_tokens=True)
                            all_tokens.extend(tokens)
                    except json.JSONDecodeError:
                        continue
            else:
                content = f.read()
                tokens = tokenizer.encode(content, add_special_tokens=True)
                all_tokens.extend(tokens)

    # Convert to tensor and save
    token_tensor = torch.tensor(all_tokens, dtype=torch.long)
    torch.save(token_tensor, output_path)

    return len(all_tokens)


def create_instruction_format(
    instruction: str,
    input_text: str = "",
    output: str = "",
) -> str:
    """Format data into instruction-tuning format."""
    if input_text:
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    return f"""### Instruction:
{instruction}

### Response:
{output}"""


def create_chat_format(messages: List[Dict[str, str]]) -> str:
    """Convert messages to chat format."""
    formatted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted.append(f"<|{role}|>\n{content}")
    return "\n".join(formatted) + "\n<|assistant|>\n"


def tokenize_function(examples: Dict[str, Any], tokenizer, max_length: int = 2048):
    """Tokenize examples for training."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def prepare_sft_dataset(
    input_path: str,
    output_path: str,
    format_type: str = "instruction",  # "instruction" or "chat"
) -> int:
    """
    Prepare dataset for supervised fine-tuning.

    Returns number of processed examples.
    """
    count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Preparing SFT data"):
            try:
                data = json.loads(line.strip())

                if format_type == "instruction":
                    text = create_instruction_format(
                        instruction=data.get("instruction", ""),
                        input_text=data.get("input", ""),
                        output=data.get("output", ""),
                    )
                elif format_type == "chat":
                    messages = data.get("messages", [])
                    text = create_chat_format(messages)
                else:
                    text = data.get("text", "")

                if text:
                    fout.write(json.dumps({"text": text}) + "\n")
                    count += 1

            except json.JSONDecodeError:
                continue

    return count


def split_dataset(
    input_path: str,
    train_path: str,
    val_path: str,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple:
    """
    Split dataset into train and validation sets.

    Returns (train_count, val_count).
    """
    import random
    random.seed(seed)

    # Read all lines
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Shuffle
    random.shuffle(lines)

    # Split
    val_size = int(len(lines) * val_ratio)
    val_lines = lines[:val_size]
    train_lines = lines[val_size:]

    # Write
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    return len(train_lines), len(val_lines)
