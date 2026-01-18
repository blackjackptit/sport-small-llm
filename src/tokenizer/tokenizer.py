"""Custom BPE tokenizer for sports domain."""

from pathlib import Path
from typing import List, Optional, Iterator
import json

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence


class SportsTokenizer:
    """Custom BPE tokenizer optimized for sports domain text."""

    SPECIAL_TOKENS = [
        "<pad>",    # Padding token
        "<s>",      # Beginning of sequence
        "</s>",     # End of sequence
        "<unk>",    # Unknown token
        "<mask>",   # Mask token for MLM (optional)
    ]

    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        lowercase: bool = False,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.lowercase = lowercase
        self.tokenizer = None

    def _create_tokenizer(self) -> Tokenizer:
        """Create a new BPE tokenizer."""
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # Normalizer
        normalizers = [NFD(), StripAccents()]
        if self.lowercase:
            normalizers.append(Lowercase())
        tokenizer.normalizer = Sequence(normalizers)

        # Pre-tokenizer (split on whitespace and punctuation)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        # Decoder
        tokenizer.decoder = decoders.ByteLevel()

        # Post-processor (add special tokens)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        return tokenizer

    def train(
        self,
        files: List[str] = None,
        text_iterator: Iterator[str] = None,
        show_progress: bool = True,
    ):
        """
        Train the tokenizer on a corpus.

        Args:
            files: List of file paths to train on
            text_iterator: Iterator yielding text strings
            show_progress: Show training progress
        """
        self.tokenizer = self._create_tokenizer()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=show_progress,
        )

        if files:
            self.tokenizer.train(files, trainer)
        elif text_iterator:
            self.tokenizer.train_from_iterator(text_iterator, trainer)
        else:
            raise ValueError("Must provide either files or text_iterator")

        # Set special token IDs
        self._setup_special_tokens()

    def _setup_special_tokens(self):
        """Configure special token IDs."""
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.bos_token_id = self.tokenizer.token_to_id("<s>")
        self.eos_token_id = self.tokenizer.token_to_id("</s>")
        self.unk_token_id = self.tokenizer.token_to_id("<unk>")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
    ) -> List[int]:
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")

        # Enable truncation/padding
        if truncation and max_length:
            self.tokenizer.enable_truncation(max_length=max_length)
        if padding and max_length:
            self.tokenizer.enable_padding(
                pad_id=self.pad_token_id,
                pad_token="<pad>",
                length=max_length,
            )

        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

        return encoding.ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
    ) -> List[List[int]]:
        """Encode a batch of texts."""
        if truncation and max_length:
            self.tokenizer.enable_truncation(max_length=max_length)
        if padding and max_length:
            self.tokenizer.enable_padding(
                pad_id=self.pad_token_id,
                pad_token="<pad>",
                length=max_length,
            )

        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]

    def batch_decode(self, batch_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token IDs."""
        return self.tokenizer.decode_batch(batch_ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str):
        """Save tokenizer to file."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        self.tokenizer.save(str(path / "tokenizer.json"))

        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "lowercase": self.lowercase,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SportsTokenizer":
        """Load tokenizer from file."""
        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        tokenizer = cls(
            vocab_size=config["vocab_size"],
            min_frequency=config["min_frequency"],
            lowercase=config["lowercase"],
        )

        # Load tokenizer
        tokenizer.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

        # Set special token IDs
        tokenizer.pad_token_id = config["pad_token_id"]
        tokenizer.bos_token_id = config["bos_token_id"]
        tokenizer.eos_token_id = config["eos_token_id"]
        tokenizer.unk_token_id = config["unk_token_id"]

        return tokenizer

    @property
    def vocab_size_actual(self) -> int:
        """Get actual vocabulary size after training."""
        if self.tokenizer is None:
            return 0
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> dict:
        """Get vocabulary as dict."""
        if self.tokenizer is None:
            return {}
        return self.tokenizer.get_vocab()
