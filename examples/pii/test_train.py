"""
Quick smoke test for PII training — hardcoded tiny dataset, no HF downloads.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

from datasets import Dataset
from mlx_trainer.args import TrainingArgs
from mlx_trainer.load import load_token_classification
from mlx_trainer.token_classification_trainer import TokenClassificationTrainer

LABELS = [
    "O", "B-PER", "I-PER", "B-EMAIL", "I-EMAIL",
    "B-PHONE", "I-PHONE", "B-ADDRESS", "I-ADDRESS",
]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

# ── Tiny hardcoded dataset ────────────────────────────────────────
TOKENS_AND_LABELS = [
    # "Alice lives in New York"
    (["Alice", "lives", "in", "New", "York"],
     ["B-PER", "O", "O", "B-ADDRESS", "I-ADDRESS"]),
    # "Bob's email is bob@test.com"
    (["Bob", "'s", "email", "is", "bob", "@", "test", ".", "com"],
     ["B-PER", "O", "O", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL"]),
    # "Call 555-1234 please"
    (["Call", "555", "-", "1234", "please"],
     ["O", "B-PHONE", "I-PHONE", "I-PHONE", "O"]),
    # "Hi Carol from London"
    (["Hi", "Carol", "from", "London"],
     ["O", "B-PER", "O", "B-ADDRESS"]),
    # "My phone is 555-9876"
    (["My", "phone", "is", "555", "-", "9876"],
     ["O", "O", "O", "B-PHONE", "I-PHONE", "I-PHONE"]),
]

rows = [{"tokens": t, "labels": l} for t, l in TOKENS_AND_LABELS]
# Duplicate to get enough samples for train+eval split
rows = rows * 10
train_dataset = Dataset.from_list(rows[:40])
eval_dataset = Dataset.from_list(rows[40:])

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ── Load model ────────────────────────────────────────────────────
print("\n[1/3] Loading model...")
model, tokenizer = load_token_classification(
    "answerdotai/ModernBERT-large",
    train=True,
    num_labels=len(LABELS),
    id2label=id2label,
)

# ── Training args (tiny) ──────────────────────────────────────────
training_args = TrainingArgs(
    output_dir="outputs/modernbert_pii_test",
    learning_rate=5e-5,
    num_train_epochs=2,
    batch_size=2,
    gradient_accumulation_steps=1,
    logging_steps=2,
    eval_steps=5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_length=64,
    grad_checkpoint=False,
)

# ── Train ─────────────────────────────────────────────────────────
print("\n[2/3] Training...")
class_weights = [0.2] * len(LABELS)  # downweight O
for k in ["B-PER", "I-PER", "B-EMAIL", "I-EMAIL", "B-PHONE", "I-PHONE", "B-ADDRESS", "I-ADDRESS"]:
    class_weights[label2id[k]] = 1.0

trainer = TokenClassificationTrainer(
    model=model,
    tokenizer=tokenizer,
    training_args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    id2label=id2label,
    label2id=label2id,
    class_weights=class_weights,
)

trainer.train()
print("\n✓ Smoke test passed — training runs end-to-end.")
