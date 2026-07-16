"""Collator for token classification: aligns BIO labels to subword tokens."""

import logging
from typing import Any, Dict, List

import mlx.core as mx

logger = logging.getLogger(__name__)


class TokenClassificationCollator:
    """Batch collator for token classification with BIO label alignment.

    Accepts either:
      - {"tokens": List[List[str]], "labels": List[List[str]]}
      - {"text": List[str], "bio_tags": List[List[str]]}

    Returns:
        {"input_ids", "attention_mask", "labels"} as mlx arrays.

    Label alignment rules:
      - word_id is None ([CLS], [SEP], PAD) → -100
      - First subword of a word → original label
      - Subsequent subword → I-XXX continuation of same entity
    """

    def __init__(self, tokenizer: Any, max_length: int = 512, label2id: Dict[str, int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or {}

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        # Determine input format
        if "tokens" in features[0]:
            batch_words = [f["tokens"] for f in features]
            batch_labels = [f["labels"] for f in features]
        elif "text" in features[0]:
            # "text" format: need to split on whitespace before tokenizing
            batch_words = [f["text"].split() for f in features]
            batch_labels = [f["bio_tags"] for f in features]
        else:
            raise ValueError(
                "Features must contain either 'tokens'+'labels' or 'text'+'bio_tags'"
            )

        # Tokenize each example separately to get per-example word_ids()
        all_input_ids = []
        all_attention_mask = []
        all_labels = []

        for words, labels in zip(batch_words, batch_labels):
            encoding = self.tokenizer(
                words,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                is_split_into_words=True,
            )

            word_ids = encoding.word_ids()
            aligned_labels = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    # Special token: [CLS], [SEP], [PAD]
                    aligned_labels.append(-100)
                elif word_id != previous_word_id:
                    # First subword of this word → original label
                    if word_id < len(labels):
                        label_str = labels[word_id]
                        aligned_labels.append(self.label2id[label_str])
                    else:
                        aligned_labels.append(-100)
                else:
                    # Subsequent subword of same word → I- continuation
                    if word_id < len(labels):
                        original_label = labels[word_id]
                        aligned_labels.append(self._to_continuation_id(original_label))
                    else:
                        aligned_labels.append(-100)

                previous_word_id = word_id

            all_input_ids.append(encoding["input_ids"])
            all_attention_mask.append(encoding["attention_mask"])
            all_labels.append(aligned_labels)

        # Pad to max length in batch
        max_len = max(len(ids) for ids in all_input_ids)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                logger.warning(
                    "No pad_token defined. Using eos_token_id=%d for padding.",
                    self.tokenizer.eos_token_id,
                )
                pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError("No pad_token or eos_token available for padding.")

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, labs in zip(all_input_ids, all_attention_mask, all_labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)

        return {
            "input_ids": mx.array(padded_input_ids, dtype=mx.int32),
            "attention_mask": mx.array(padded_attention_mask, dtype=mx.int32),
            "labels": mx.array(padded_labels, dtype=mx.int32),
        }

    def _to_continuation_id(self, label_str: str) -> int:
        """Get the label ID for the I- continuation of a label string."""
        if label_str == "O":
            return self.label2id.get("O", 0)
        if label_str.startswith("B-"):
            i_label = "I-" + label_str[2:]
            return self.label2id.get(i_label, self.label2id.get(label_str, 0))
        # Already an I- label or unknown — keep as-is
        return self.label2id.get(label_str, 0)
