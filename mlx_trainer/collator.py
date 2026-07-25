import logging
import mlx.core as mx
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class TextClassificationCollator:
    tokenizer: Any
    max_length: int = 512

    def __call__(self, features: Any) -> Dict[str, mx.array]:
        if self.tokenizer.pad_token_id is None:
            # Check if model config defines a pad token
            pad_token = getattr(self.tokenizer, "pad_token", None)
            if pad_token is None:
                # Fall back to eos_token with warning
                if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
                    logger.warning(
                        "No pad_token defined. Falling back to eos_token='%s' for padding. "
                        "This may not be ideal for all models.",
                        self.tokenizer.eos_token,
                    )
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    raise ValueError(
                        "No pad_token or eos_token available. "
                        "Please set tokenizer.pad_token explicitly."
                    )

        # Row-oriented list → column-oriented dict
        if isinstance(features, list):
            features = {k: [f[k] for f in features] for k in features[0]}

        batch = self.tokenizer(
            features["text"],
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="mlx",
        )
        batch["labels"] = mx.array(features["label"], dtype=mx.int32)
        return dict(batch)
