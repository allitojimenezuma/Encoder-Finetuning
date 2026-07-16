"""
ModelArgs for ModernBERT — clean dataclass with proper annotated fields.

Replaces mlx_raclate.models.modernbert.ModelArgs.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class ModelArgs:
    """Configuration for ModernBERT base model + classification heads."""

    # --- architecture ---
    model_type: str = "modernbert"
    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12

    # --- attention ---
    attention_bias: bool = False
    attention_dropout: float = 0.0
    global_attn_every_n_layers: int = 3
    global_rope_theta: float = 160000.0
    local_attention: int = 128
    local_rope_theta: float = 10000.0

    # --- position / tokens ---
    max_position_embeddings: int = 8192
    bos_token_id: int = 50281
    cls_token_id: int = 50281
    eos_token_id: int = 50282
    sep_token_id: int = 50282
    pad_token_id: int = 50368

    # --- embeddings ---
    embedding_dropout: float = 0.0

    # --- norm ---
    norm_eps: float = 1e-5
    norm_bias: bool = False

    # --- MLP / FFN ---
    mlp_bias: bool = False
    mlp_dropout: float = 0.0

    # --- output ---
    output_hidden_states: bool = False

    # --- initializer ---
    initializer_range: float = 0.02
    initializer_cutoff_factor: float = 2.0

    # --- prediction heads (MLM) ---
    decoder_bias: bool = True
    sparse_prediction: bool = True
    sparse_pred_ignore_index: int = -100

    # --- classification heads ---
    classifier_pooling: Literal["cls", "mean"] = "cls"
    classifier_dropout: float = 0.0
    classifier_bias: bool = False

    # --- task flags ---
    is_regression: Optional[bool] = None
    label2id: Optional[Dict[str, int]] = None
    id2label: Optional[Dict[int, str]] = None
    pipeline_config: Optional[Dict[str, Any]] = None
    use_late_interaction: bool = False

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        """Create from a config dict, ignoring unknown keys."""
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    @property
    def num_labels(self) -> int:
        """
        Number of classification labels:
        - regression → 1
        - binary sigmoid (via pipeline_config) → 1
        - otherwise → len(id2label)
        """
        if self.is_regression:
            return 1
        if self.pipeline_config and self.pipeline_config.get("binary_sigmoid", False):
            return 1
        if self.id2label is None:
            raise ValueError(
                "id2label mapping must be provided for categorical classification. "
                "For regression or binary classification with sigmoid output, "
                "set is_regression=True or binary_sigmoid=True in pipeline_config."
            )
        return len(self.id2label)
