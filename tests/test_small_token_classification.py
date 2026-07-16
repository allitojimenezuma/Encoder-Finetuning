"""Smoke test: fp16 load + fp16 training on tiny ModernBERT token classification."""
import sys
sys.path.insert(0, ".")

import mlx.core as mx
from mlx.utils import tree_flatten
from datasets import Dataset
from mlx_trainer import TokenClassificationTrainer, TrainingArgs, load_token_classification

LABEL_CFG = {
    "num_labels": 5,
    "id2label": {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-EMAIL", 4: "I-EMAIL"},
    "label2id": {"O": 0, "B-PER": 1, "I-PER": 2, "B-EMAIL": 3, "I-EMAIL": 4},
}

# --- Test 1: fp16 load ---
print("Test 1: fp16 load")
model_fp16, _ = load_token_classification(
    "answerdotai/ModernBERT-large", num_labels=5,
    id2label=LABEL_CFG["id2label"],
)
params = dict(tree_flatten(model_fp16.parameters()))
assert params[list(params.keys())[0]].dtype == mx.float16, "Not fp16!"
assert model_fp16.num_labels == 5, "Wrong num_labels!"
print("  ✓ fp16 OK\n")
del model_fp16

# --- Test 2: fp16 training ---
print("Test 2: fp16 training")
# Tiny synthetic token classification dataset (tokens + BIO labels)
samples = [
    {"tokens": ["Alice", "lives", "in", "Paris"], "labels": ["B-PER", "O", "O", "O"]},
    {"tokens": ["Bob", "emails", "alice", "@", "test", ".", "com"], "labels": ["B-PER", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL"]},
    {"tokens": ["Charlie", "met", "Diana", "today"], "labels": ["B-PER", "O", "B-PER", "O"]},
    {"tokens": ["Send", "email", "to", "bob", "@", "mail", ".", "org"], "labels": ["O", "O", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL"]},
    {"tokens": ["Eve", "and", "Frank", "are", "friends"], "labels": ["B-PER", "O", "B-PER", "O", "O"]},
    {"tokens": ["Grace", "sent", "mail", "to", "eve", "@", "test", ".", "io"], "labels": ["B-PER", "O", "O", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL"]},
    {"tokens": ["Hank", "called", "Alice"], "labels": ["B-PER", "O", "B-PER"]},
    {"tokens": ["Write", "to", "charlie", "@", "dev", ".", "net"], "labels": ["O", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL"]},
    {"tokens": ["Ivy", "met", "Bob", "and", "Grace"], "labels": ["B-PER", "O", "B-PER", "O", "B-PER"]},
    {"tokens": ["Jack", "emailed", "frank", "@", "mail", ".", "com"], "labels": ["B-PER", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL"]},
]
# Duplicate for enough training steps
data = samples * 4
ds = Dataset.from_dict({
    "tokens": [s["tokens"] for s in data],
    "labels": [s["labels"] for s in data],
})

model, tokenizer = load_token_classification(
    "answerdotai/ModernBERT-large",
    train=True, dtype=mx.float16,
    num_labels=5,
    id2label=LABEL_CFG["id2label"],
)

# Class weights: O=1, entities=5 (address class imbalance)
class_weights = [1.0, 5.0, 5.0, 5.0, 5.0]

args = TrainingArgs(
    output_dir="outputs/test_small_token_classification",
    num_train_epochs=2, batch_size=4,
    gradient_accumulation_steps=1, logging_steps=1, save_steps=999,
    save_total_limit=None, warmup_ratio=0.0, lr_scheduler_type="constant",
    learning_rate=5e-5, max_length=128,
)

trainer = TokenClassificationTrainer(
    model=model, tokenizer=tokenizer, training_args=args,
    train_dataset=ds, eval_dataset=ds,
    id2label=LABEL_CFG["id2label"],
    label2id=LABEL_CFG["label2id"],
    class_weights=class_weights,
)
trainer.train()
print("\n✓ All tests passed")
