"""
Pure-MLX ModernBERT implementation for sequence classification.

Replaces mlx_raclate.models.modernbert with a self-contained module.
Architecture: embeddings → RoPE attention (local + global) → GLU MLP → classification head.
"""

from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .modernbert_config import ModelArgs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_pooling(
    token_embeddings: mx.array, attention_mask: mx.array
) -> mx.array:
    """Average non-pad token embeddings."""
    input_mask_expanded = mx.expand_dims(attention_mask, -1)
    input_mask_expanded = mx.broadcast_to(
        input_mask_expanded, token_embeddings.shape
    ).astype(mx.float32)
    sum_embeddings = mx.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = mx.maximum(mx.sum(input_mask_expanded, axis=1), 1e-9)
    return sum_embeddings / sum_mask


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.drop = nn.Dropout(p=config.embedding_dropout)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.tok_embeddings(input_ids)
        x = self.norm(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# MLP / FFN (GLU)
# ---------------------------------------------------------------------------

class ModernBertMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size, config.intermediate_size * 2, bias=config.mlp_bias
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        x = self.Wi(hidden_states)
        split = x.shape[-1] // 2
        inp, gate = x[..., :split], x[..., split:]
        return self.Wo(self.drop(self.act(inp) * gate))


# ---------------------------------------------------------------------------
# Self-attention (global + local sliding window, RoPE)
# ---------------------------------------------------------------------------

class ModernBertAttention(nn.Module):
    def __init__(self, config: ModelArgs, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads

        self.Wqkv = nn.Linear(
            config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias
        )

        # Local vs global attention
        if layer_id is not None and layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (
                config.local_attention // 2,
                config.local_attention // 2,
            )
        else:
            self.local_attention = (-1, -1)

        rope_theta = config.global_rope_theta
        if self.local_attention != (-1, -1) and config.local_rope_theta is not None:
            rope_theta = config.local_rope_theta

        self.rotary_emb = nn.RoPE(dims=self.head_dim, base=rope_theta)

        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.out_drop = (
            nn.Dropout(p=config.attention_dropout)
            if config.attention_dropout > 0.0
            else nn.Identity()
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        sliding_window_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> tuple:
        batch_size = hidden_states.shape[0]
        qkv = mx.reshape(
            self.Wqkv(hidden_states),
            (batch_size, -1, 3, self.num_heads, self.head_dim),
        )
        qkv = mx.transpose(qkv, [0, 3, 2, 1, 4])
        query, key, value = mx.split(qkv, indices_or_sections=3, axis=2)
        query = query.squeeze(2)
        key = key.squeeze(2)
        value = value.squeeze(2)

        query = self.rotary_emb(query)
        key = self.rotary_emb(key)

        # Choose mask
        if self.local_attention != (-1, -1):
            mask = sliding_window_mask
        else:
            mask = attention_mask

        scale = query.shape[-1] ** -0.5
        attn_output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=scale, mask=mask
        )

        attn_output = mx.transpose(attn_output, [0, 2, 1, 3])
        attn_output = mx.reshape(attn_output, (batch_size, -1, self.all_head_size))

        hidden_states = self.Wo(attn_output)
        hidden_states = self.out_drop(hidden_states)
        return (hidden_states,)


# ---------------------------------------------------------------------------
# Encoder layer (pre-norm residual)
# ---------------------------------------------------------------------------

class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_id: Optional[int] = None):
        super().__init__()
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp = ModernBertMLP(config)
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        sliding_window_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> tuple:
        normed = self.attn_norm(hidden_states)
        attn_out = self.attn(
            normed,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + attn_out[0]
        mlp_out = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + mlp_out
        return (hidden_states,)


# ---------------------------------------------------------------------------
# Base encoder model
# ---------------------------------------------------------------------------

class ModernBertModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = [
            ModernBertEncoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ]
        self.final_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value: nn.Embedding):
        self.embeddings.tok_embeddings = value

    def _update_attention_mask(
        self, attention_mask: mx.array, model_dtype: mx.Dtype
    ) -> tuple:
        """Build global + sliding-window masks from raw padding mask."""
        batch_size, seq_len = attention_mask.shape
        neg_inf = mx.array(-1e4, dtype=model_dtype)

        # Additive padding mask: 0 for valid, -inf for pad
        additive_mask = mx.where(attention_mask == 1, 0.0, neg_inf)
        additive_mask = additive_mask[:, None, None, :]  # (B, 1, 1, S)

        # Global attention mask (broadcast over heads and query positions)
        global_mask = mx.broadcast_to(additive_mask, (batch_size, 1, seq_len, seq_len))

        # Sliding window mask
        rows = mx.arange(seq_len)[None, :]  # (1, S)
        distance = mx.abs(rows - rows.T)    # (S, S)
        window = mx.where(
            distance <= (self.config.local_attention // 2),
            mx.ones_like(distance),
            mx.zeros_like(distance),
        )
        window = window[None, None, :, :]  # (1, 1, S, S)
        window = mx.broadcast_to(window, global_mask.shape)

        sliding_mask = mx.where(window, global_mask, neg_inf)

        global_mask = global_mask.astype(model_dtype)
        sliding_mask = sliding_mask.astype(model_dtype)

        return global_mask, sliding_mask

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        sliding_window_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Dict[str, Any]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        batch_size, seq_len = input_ids.shape[:2]

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len))

        hidden_states = self.embeddings(input_ids)
        model_dtype = hidden_states.dtype

        global_mask, sliding_mask = self._update_attention_mask(
            attention_mask, model_dtype
        )

        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(
                hidden_states,
                attention_mask=global_mask,
                sliding_window_mask=sliding_mask,
                position_ids=position_ids,
            )[0]

        hidden_states = self.final_norm(hidden_states)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states] if v is not None
            )
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
        }


# ---------------------------------------------------------------------------
# Classification prediction head
# ---------------------------------------------------------------------------

class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.norm(self.act(self.dense(hidden_states)))


# ---------------------------------------------------------------------------
# Sequence classification model
# ---------------------------------------------------------------------------

class ModelForSequenceClassification(nn.Module):
    """
    ModernBERT encoder + prediction head + linear classifier.

    Outputs {"loss", "probabilities", "hidden_states"} — compatible with
    the mlx_trainer.Trainer.
    """

    hf_transformers_arch: str = "ModernBertForSequenceClassification"

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.is_regression = config.is_regression

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = nn.Dropout(p=config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    # -- output activation --------------------------------------------------

    def _process_outputs(self, logits: mx.array) -> mx.array:
        if self.is_regression:
            return logits
        elif self.num_labels == 1:
            return mx.sigmoid(logits)
        else:
            return mx.softmax(logits, axis=-1)

    # -- loss ---------------------------------------------------------------

    def _compute_loss(self, logits: mx.array, labels: mx.array) -> mx.array:
        if self.is_regression:
            return nn.losses.mse_loss(logits.squeeze(), labels.squeeze())
        elif self.num_labels == 1:
            return nn.losses.binary_cross_entropy(mx.sigmoid(logits), labels)
        else:
            return nn.losses.cross_entropy(
                logits.reshape(-1, self.num_labels), labels.reshape(-1)
            )

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

        # Pooling
        if self.config.classifier_pooling == "cls":
            pooled = last_hidden_state[:, 0]
        else:  # mean
            pooled = _mean_pooling(last_hidden_state, attention_mask)

        pooled = self.head(pooled)
        pooled = self.drop(pooled)
        logits = self.classifier(pooled)

        processed_logits = self._process_outputs(logits)

        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)

        if not return_dict:
            return [loss, processed_logits, outputs[1:]]

        return {
            "loss": loss,
            "probabilities": processed_logits,
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
