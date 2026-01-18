"""Dataset classes for pretraining Sports Domain LLM."""

import json
from pathlib import Path
from typing import Iterator, Optional, List, Dict
import random

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader


class PretrainDataset(Dataset):
    """Dataset for pretraining with tokenized text chunks."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        stride: int = 512,
    ):
        """
        Args:
            data_path: Path to tokenized data file (.pt)
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for overlapping chunks
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        # Load pre-tokenized data
        self.data = torch.load(data_path)

        # Create chunks with overlap
        self.chunks = self._create_chunks()

    def _create_chunks(self) -> List[torch.Tensor]:
        """Create overlapping chunks from tokenized data."""
        chunks = []
        for i in range(0, len(self.data) - self.max_length, self.stride):
            chunk = self.data[i : i + self.max_length]
            chunks.append(chunk)
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]

        # For causal LM, input and target are shifted
        input_ids = chunk[:-1]
        labels = chunk[1:]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class StreamingPretrainDataset(IterableDataset):
    """Streaming dataset for large-scale pretraining."""

    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        max_length: int = 2048,
        shuffle_files: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            data_files: List of JSONL file paths
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            shuffle_files: Whether to shuffle file order
            seed: Random seed
        """
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_files = shuffle_files
        self.seed = seed

    def _read_file(self, file_path: str) -> Iterator[str]:
        """Read text from a JSONL file."""
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = data.get("text", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue

    def _tokenize_and_chunk(self, text: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Tokenize text and create training chunks."""
        # Tokenize with BOS/EOS
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # Skip if too short
        if len(token_ids) < 2:
            return

        # Create chunks
        for i in range(0, len(token_ids) - 1, self.max_length - 1):
            chunk = token_ids[i : i + self.max_length]

            # Pad if necessary
            if len(chunk) < self.max_length:
                chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_length - len(chunk))

            chunk = torch.tensor(chunk, dtype=torch.long)
            input_ids = chunk[:-1]
            labels = chunk[1:].clone()

            # Mask padding in labels
            labels[labels == self.tokenizer.pad_token_id] = -100

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        files = self.data_files.copy()
        if self.shuffle_files:
            random.seed(self.seed)
            random.shuffle(files)

        for file_path in files:
            for text in self._read_file(file_path):
                yield from self._tokenize_and_chunk(text)


class TextDataset(Dataset):
    """Simple dataset for loading raw text files."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Load data
        self._load_data(data_path)

    def _load_data(self, data_path: str):
        """Load data from file."""
        path = Path(data_path)

        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    text = data.get("text", "")
                    if text:
                        self.samples.append(text)
        elif path.suffix == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                # Split into paragraphs
                paragraphs = content.split("\n\n")
                self.samples.extend([p.strip() for p in paragraphs if p.strip()])
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        text = item.get("text", "") if isinstance(item, dict) else str(item)
                        if text:
                            self.samples.append(text)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]

        # Tokenize
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )

        token_ids = torch.tensor(token_ids, dtype=torch.long)

        # Pad to max_length
        if len(token_ids) < self.max_length:
            padding = torch.full(
                (self.max_length - len(token_ids),),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            token_ids = torch.cat([token_ids, padding])

        input_ids = token_ids[:-1]
        labels = token_ids[1:].clone()

        # Mask padding in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
