"""
ModernBERT loader — fp16 by default.
Self-contained model classes (pure MLX, no external model deps)."""
import glob
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from huggingface_hub import snapshot_download

from .modernbert_config import ModelArgs
from .modernbert_model import ModelForSequenceClassification
from .tokenizer_utils import load_tokenizer


def _get_model_path(repo_id: str) -> Path:
    path = Path(repo_id)
    if path.exists():
        return path
    return Path(snapshot_download(
        repo_id=repo_id,
        allow_patterns=["*.json", "*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "*.txt"],
    ))


def _init_head_weights(model: nn.Module, weights: dict, config: Any, dtype: mx.Dtype):
    """Initialize weights that are in the model but missing from the checkpoint."""
    model_params = dict(tree_flatten(model.parameters()))
    init_range = getattr(config, "initializer_range", 0.02)
    count = 0

    for key, param in model_params.items():
        if key in weights:
            continue
        # Initialize any missing weight (not just head-specific)
        if "bias" in key:
            weights[key] = mx.zeros(param.shape, dtype=dtype)
        elif "norm" in key or "ln" in key or "layernorm" in key.lower():
            weights[key] = mx.ones(param.shape, dtype=dtype)
        else:
            weights[key] = mx.random.normal(param.shape, scale=init_range, dtype=dtype)
        count += 1

    if count:
        print(f"[load] Initialized {count} missing weights ({dtype})")


def load(
    repo_id: str,
    model_config: Optional[Dict[str, Any]] = None,
    train: bool = False,
    dtype: mx.Dtype = mx.float16,
    tokenizer_config: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Any]:
    """
    Load ModernBERT for sequence classification.

    Args:
        repo_id: HuggingFace repo or local path.
        model_config: Extra config overrides (e.g. num_labels, id2label).
        train: If True, init missing head weights.
        dtype: Target dtype (default fp16).
        tokenizer_config: Extra tokenizer kwargs.

    Returns:
        (model, tokenizer) tuple.
    """
    model_path = _get_model_path(repo_id)

    with open(model_path / "config.json") as f:
        config = json.load(f)

    if model_config:
        config.update(model_config)

    model_args = ModelArgs.from_dict(config)
    model = ModelForSequenceClassification(model_args)
    model.set_dtype(dtype)

    # Load + cast weights
    weights = {}
    for wf in glob.glob(str(model_path / "model*.safetensors")):
        weights.update(mx.load(wf))
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    _init_head_weights(model, weights, model_args, dtype)

    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    model.eval()

    tokenizer = load_tokenizer(model_path, tokenizer_config or {})
    print(f"[load] {repo_id} @ {dtype}")
    return model, tokenizer
