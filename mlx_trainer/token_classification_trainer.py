"""
Token classification trainer with entity-level evaluation.

Extends the base Trainer with:
  - TokenClassificationCollator for BIO label alignment
  - Entity-level P/R/F1 via seqeval
  - Weighted cross-entropy for class imbalance
"""

import gc
from functools import partial
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np
from mlx.utils import tree_map

try:
    from seqeval.metrics import classification_report as seqeval_report
except ImportError:
    seqeval_report = None

from .token_classification_collator import TokenClassificationCollator
from .trainer import Trainer


class TokenClassificationTrainer(Trainer):
    """Trainer for token-level classification (NER / PII detection).

    Adds to the base Trainer:
      - TokenClassificationCollator with BIO label alignment
      - Entity-level evaluation via seqeval (P/R/F1 per entity type)
      - Optional class-weighted cross-entropy to handle O vs entity imbalance
    """

    def __init__(
        self,
        model,
        tokenizer,
        training_args,
        train_dataset,
        eval_dataset=None,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        class_weights: Optional[List[float]] = None,
    ):
        self.id2label = id2label or {}
        self.label2id = label2id or {}
        self._class_weights = class_weights

        # Parent init: sets up optimizer, compiled fns, default collator, etc.
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Override collator for token classification
        self.collator = TokenClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=training_args.max_length,
            label2id=self.label2id,
        )

        # Recompile training/eval with weighted loss if weights provided
        if class_weights is not None:
            self._recompile_weighted_loss(class_weights)

    # -- weighted loss recompilation ----------------------------------------

    def _recompile_weighted_loss(self, class_weights: List[float]):
        """Recompile step_fn, update_fn, eval_fn with class-weighted CE."""
        weights = mx.array(class_weights, dtype=mx.float32)
        num_labels = self.model.num_labels

        def weighted_loss_fn(model, batch):
            model_input = {k: v for k, v in batch.items() if k != "labels"}
            out = model(**model_input)
            logits = out["logits"]
            labels = batch["labels"]

            valid_mask = labels != -100
            valid_count = mx.sum(valid_mask.astype(mx.float32))
            valid_count = mx.maximum(valid_count, mx.array(1.0))

            flat_logits = logits.reshape(-1, num_labels)
            flat_labels = labels.reshape(-1)
            safe_labels = mx.maximum(flat_labels, 0)

            ce = nn.losses.cross_entropy(flat_logits, safe_labels)

            # Per-class weighting
            flat_weights = weights[safe_labels]
            weighted_ce = ce * flat_weights

            masked = mx.where(
                valid_mask.reshape(-1), weighted_ce, mx.zeros_like(weighted_ce)
            )
            return mx.sum(masked) / valid_count

        grad_fn = nn.value_and_grad(self.model, weighted_loss_fn)

        step_state = [self.model.state, mx.random.state]

        @partial(mx.compile, inputs=step_state, outputs=step_state)
        def step_fn(batch):
            loss, grads = grad_fn(self.model, batch)
            return loss, grads

        update_state = [self.model.state, self.optimizer.state, mx.random.state]

        @partial(mx.compile, inputs=update_state, outputs=update_state)
        def update_fn(accumulated_grads, scale):
            scaled = tree_map(
                lambda g: (g * scale).astype(mx.float32), accumulated_grads
            )
            clipped, _ = opt.clip_grad_norm(scaled, self.args.max_grad_norm)
            self.optimizer.update(self.model, clipped)
            return None

        eval_state = [self.model.state]

        @partial(mx.compile, inputs=eval_state, outputs=eval_state)
        def eval_fn(batch):
            # NOTE: Eval loss is *unweighted* CE (model's internal loss).
            # Training loss uses class-weighted CE (see weighted_loss_fn).
            # The two scales are intentionally different:
            # unweighted eval loss is more interpretable as a raw probability metric,
            # while weighted training loss upweights minority entity classes.
            out = self.model(**batch)
            # out["loss"] is already a mean over valid tokens; no need for mx.mean()
            loss = out["loss"].astype(mx.float32)
            preds = mx.argmax(out["probabilities"], axis=-1)
            return loss, preds

        self.step_fn = step_fn
        self.update_fn = update_fn
        self.eval_fn = eval_fn

    # -- entity-level evaluation -------------------------------------------

    def evaluate(self):
        """Evaluate with entity-level metrics (seqeval) + token accuracy."""
        if seqeval_report is None:
            raise ImportError(
                "seqeval is required for entity-level evaluation. "
                "Install with: pip install seqeval"
            )

        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        total_correct = 0
        total_valid = 0

        all_y_true: List[List[str]] = []
        all_y_pred: List[List[str]] = []

        gc.collect()
        mx.clear_cache()

        for raw_batch in self._batches(
            self.eval_dataset, self.args.eval_batch_size
        ):
            batch = self.collator(raw_batch)
            labels = batch["labels"]

            loss, preds = self.eval_fn(batch)
            mx.eval(loss, preds)

            total_loss += loss.item()
            n_batches += 1

            preds_np = np.array(preds)
            labels_np = np.array(labels)

            # Per-sequence: filter -100, convert to BIO strings
            for i in range(preds_np.shape[0]):
                mask = labels_np[i] != -100
                seq_labels = labels_np[i][mask]
                seq_preds = preds_np[i][mask]

                total_correct += int(np.sum(seq_preds == seq_labels))
                total_valid += int(np.sum(mask))

                all_y_true.append(
                    [self.id2label[int(l)] for l in seq_labels]
                )
                all_y_pred.append(
                    [self.id2label[int(p)] for p in seq_preds]
                )

            del loss, preds, batch
            mx.clear_cache()

        # -- metrics -------------------------------------------------------
        eval_loss = total_loss / max(n_batches, 1)
        accuracy = total_correct / max(total_valid, 1)

        report_dict = seqeval_report(
            all_y_true, all_y_pred, mode="strict", output_dict=True
        )
        report_str = seqeval_report(
            all_y_true, all_y_pred, mode="strict"
        )

        # Micro-averaged entity metrics
        micro = report_dict.get("micro avg", {})
        entity_p = micro.get("precision", 0.0)
        entity_r = micro.get("recall", 0.0)
        entity_f1 = micro.get("f1-score", 0.0)

        metrics = {
            "eval_loss": eval_loss,
            "accuracy": accuracy,
            "entity_precision": float(entity_p),
            "entity_recall": float(entity_r),
            "entity_f1": float(entity_f1),
        }

        # Per-entity-type F1
        for label_name, scores in report_dict.items():
            if isinstance(scores, dict):
                metrics[f"entity_f1_{label_name}"] = scores.get(
                    "f1-score", 0.0
                )
                metrics[f"entity_precision_{label_name}"] = scores.get(
                    "precision", 0.0
                )
                metrics[f"entity_recall_{label_name}"] = scores.get(
                    "recall", 0.0
                )

        print(
            f"  Eval — loss: {eval_loss:.4f} | "
            f"acc: {accuracy:.4f} | "
            f"entity-F1: {entity_f1:.4f}"
        )
        print(f"\n{report_str}")

        return metrics
