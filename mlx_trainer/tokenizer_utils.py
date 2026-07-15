"""Minimal tokenizer wrapper — no raclate dependency."""
import json
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import AutoTokenizer


class Tokenizer:
    """Wraps a HuggingFace tokenizer for use with the trainer."""

    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer

    def save_pretrained(self, path: str):
        self._tokenizer.save_pretrained(path)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def load_tokenizer(
    model_path: Path,
    tokenizer_config: Optional[Dict[str, Any]] = None,
) -> Tokenizer:
    """Load tokenizer from model directory."""
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path, **(tokenizer_config or {}))
    return Tokenizer(hf_tokenizer)
