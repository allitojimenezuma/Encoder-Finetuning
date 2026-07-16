# mlx_trainer

Pure MLX training framework for ModernBERT. No PyTorch. No TensorFlow. Just Metal.

Fine-tune ModernBERT-large on Apple Silicon for sequence classification (text classification) and token classification (NER / PII detection) with fp16, gradient checkpointing, and entity-level evaluation.

## Install

```bash
# With uv (recommended)
uv add mlx-trainer

# With pip
pip install mlx-trainer
```

Or install from source:

```bash
uv sync
```

Requires macOS with Apple Silicon (M1/M2/M3/M4).

## Quick Start

### Sequence Classification (Binary / Multi-class)

```python
from datasets import Dataset
from mlx_trainer import Trainer, TrainingArgs
from mlx_trainer.load import load

# Load model + tokenizer
model, tokenizer = load(
    "answerdotai/ModernBERT-large",
    train=True,
    model_config={
        "num_labels": 2,
        "id2label": {0: "SAFE", 1: "INJECTION"},
        "label2id": {"SAFE": 0, "INJECTION": 1},
    },
)

# Your dataset with "text" and "label" columns
train_dataset = Dataset.from_dict({
    "text": ["Hello world", "Ignore instructions", "Normal text"],
    "label": [0, 1, 0],
})

# Train
args = TrainingArgs(
    output_dir="outputs/my_model",
    num_train_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    training_args=args,
    train_dataset=train_dataset,
)
trainer.train()
```

### Token Classification (NER / PII Detection)

```python
from datasets import Dataset
from mlx_trainer import TokenClassificationTrainer, TrainingArgs
from mlx_trainer.load import load_token_classification

# Label schema (BIO format)
LABELS = ["O", "B-PER", "I-PER", "B-EMAIL", "I-EMAIL"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

# Load model for token classification
model, tokenizer = load_token_classification(
    "answerdotai/ModernBERT-large",
    train=True,
    num_labels=len(LABELS),
    id2label=id2label,
)

# Dataset with "tokens" and "labels" columns
train_dataset = Dataset.from_dict({
    "tokens": [["Alice", "lives", "in", "Paris"], ["Bob", "emails", "alice@test.com"]],
    "labels": [["B-PER", "O", "O", "O"], ["B-PER", "O", "B-EMAIL", "I-EMAIL"]],
})

# Weighted loss for class imbalance (O class downweighted)
class_weights = [0.2, 1.0, 1.0, 1.0, 1.0]

args = TrainingArgs(
    output_dir="outputs/pii_model",
    num_train_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
)

trainer = TokenClassificationTrainer(
    model=model,
    tokenizer=tokenizer,
    training_args=args,
    train_dataset=train_dataset,
    id2label=id2label,
    label2id=label2id,
    class_weights=class_weights,
)
trainer.train()
```

## Examples

### Sequence Classification

Train a prompt injection guardrail:

```bash
python examples/sequence_classification/train.py
```

Run inference on a trained guardrail:

```bash
python examples/sequence_classification/inference.py
```

### Token Classification (PII Detection)

Train a PII detector:

```bash
python examples/pii/train_pii.py
python examples/pii/train_pii.py --epochs 5 --lr 5e-5 --o-weight 0.1
```

Run PII inference:

```bash
python examples/pii/inference_pii.py --text "My name is Alice and my email is alice@example.com"
python examples/pii/inference_pii.py --file document.txt --model outputs/modernbert_pii/checkpoint-3941
```

## Architecture

```
mlx_trainer/
├── args.py                              # TrainingArgs configuration
├── collator.py                          # TextClassificationCollator
├── load.py                              # Model loaders (sequence + token classification)
├── modernbert_config.py                 # ModelArgs dataclass
├── modernbert_model.py                  # ModernBERT + SequenceClassification head
├── token_classification_collator.py     # BIO label alignment via word_ids()
├── token_classification_model.py        # ModernBERT + TokenClassification head
├── token_classification_trainer.py      # Trainer with entity-level eval (seqeval)
├── tokenizer_utils.py                   # HF tokenizer wrapper
├── trainer.py                           # Base trainer (gradient accumulation, checkpointing)
├── train.py                             # Sequence classification training script
└── inference.py                         # Sequence classification inference script
```

### Key Features

- **Pure MLX** — No PyTorch dependency. Runs on Metal GPU.
- **fp16 training** — 50% memory reduction, no accuracy loss.
- **Gradient checkpointing** — Trade compute for memory on long sequences.
- **Entity-level evaluation** — Uses `seqeval` for proper NER metrics (P/R/F1 per entity type).
- **Weighted cross-entropy** — Handle class imbalance (e.g., 95% "O" tokens in NER).
- **Subword label alignment** — Correctly maps BIO labels through tokenizer subwords via `word_ids()`.
- **Checkpoint rotation** — Automatically keeps only N most recent checkpoints.

## Training Tips

### Class Imbalance (Token Classification)

PII datasets are heavily imbalanced (~95% "O" tokens). Use weighted loss:

```python
# Downweight O class to 0.2, keep entity classes at 1.0
class_weights = [0.2] + [1.0] * (num_labels - 1)
```

### Gradient Checkpointing

Enable for long sequences to reduce memory:

```python
args = TrainingArgs(..., grad_checkpoint=True)
```

### Resume Training

Resume from a checkpoint:

```python
args = TrainingArgs(
    ...,
    resume_from_checkpoint="outputs/my_model/checkpoint-1000",
)
```

### Learning Rate

- Sequence classification: `1e-5` to `5e-5`
- Token classification: `1e-5` to `5e-5`
- Use warmup: `warmup_ratio=0.1` (10% of steps)

## License

MIT
