"""
Token classification model for ModernBERT (BIO labeling).

Architecture: tokens → encoder → per-token prediction head → linear → [num_labels].
No pooling — classifier operates on every token's hidden state.
"""

from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .modernbert_config import ModelArgs
from .modernbert_model import ModernBertModel, ModernBertPredictionHead


class ModelForTokenClassification(nn.Module):
    """
    ModernBERT encoder + prediction head + per-token linear classifier.

    Returns {"loss", "logits", "probabilities"} — compatible with
    the mlx_trainer.Trainer.

    Special tokens ([CLS], [SEP], [PAD]) are assigned label -100
    and excluded from the loss via ``ignore_index=-100``.
    """

    hf_transformers_arch: str = "ModernBertForTokenClassification"

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = nn.Dropout(p=config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    # -- forward ------------------------------------------------------------

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Dict[str, Any]:
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones((batch_size, seq_len))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = (
            outputs["last_hidden_state"]
            if isinstance(outputs, dict)
            else outputs[0]
        )

        # Per-token head + classifier (NO pooling)
        hidden = self.head(last_hidden_state)
        hidden = self.drop(hidden)
        logits = self.classifier(hidden)  # (batch, seq_len, num_labels)

        probabilities = mx.softmax(logits, axis=-1)

        loss = None
        if labels is not None:
            # Mask positions where label == -100 (special tokens / padding)
            valid_mask = labels != -100
            valid_count = mx.sum(valid_mask.astype(mx.float32))
            valid_count = mx.maximum(valid_count, mx.array(1.0))

            flat_logits = logits.reshape(-1, self.num_labels)
            flat_labels = labels.reshape(-1)

            # Clamp -100 to 0 so indexing is valid; masked out below
            safe_labels = mx.maximum(flat_labels, 0)

            # Per-token cross entropy, zero out ignored positions
            ce = nn.losses.cross_entropy(flat_logits, safe_labels)
            masked_ce = mx.where(valid_mask.reshape(-1), ce, mx.zeros_like(ce))
            loss = mx.sum(masked_ce) / valid_count

        if not return_dict:
            return [loss, logits, outputs[1:]]

        return {
            "loss": loss,
            "logits": logits,
            "probabilities": probabilities,
            "hidden_states": outputs.get("hidden_states", None),
        }

    # -- weight sanitization (for loading HF checkpoints) --------------------

    def sanitize(self, weights: dict) -> dict:
        """Align HF checkpoint keys with this module's parameter names."""
        sanitized = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            if k in ("decoder.bias",):
                continue
            if k.startswith("bert"):
                k = k.replace("bert.", "model.")
            sanitized[k] = v
        return sanitized
