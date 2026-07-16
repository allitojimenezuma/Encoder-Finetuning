# PLAN: Token Classification Extension for ModernBERT (PII Detection)

## Context

The current `mlx_trainer` library implements **Sequence Classification** for ModernBERT (prompt injection guardrail). We need to extend it to also support **Token Classification** for PII detection / anonymization, using BIO labeling with weighted loss.

The existing library is well-structured: `args.py`, `tokenizer_utils.py`, `modernbert_config.py`, and most of `trainer.py` are generic and reusable. The changes are surgical.

---

## Architecture Overview

```
Current (Sequence Classification):
  Tokens → Encoder → CLS pooling → Linear → [SAFE, INJECTION]

Target (Token Classification):
  Tokens → Encoder → per-token hidden states → Linear → [O, B-PER, I-PER, B-EMAIL, ...]
```

---

## Files to Create

### 1. `mlx_trainer/token_classification_model.py` (NEW)

**ModernBERT + Token Classification head.**

```
ModelForTokenClassification(nn.Module):
  - model: ModernBertModel          (reuse from modernbert_model.py)
  - head: ModernBertPredictionHead  (reuse from modernbert_model.py)
  - classifier: Linear(hidden_size, num_labels)

  __call__(input_ids, attention_mask, labels=None):
    last_hidden_state = self.model(...)   # (batch, seq_len, hidden_size)
    head_out = self.head(last_hidden_state)
    logits = self.classifier(head_out)    # (batch, seq_len, num_labels)
    loss = cross_entropy(logits, labels)  # if labels provided
    return {"loss", "logits", "probabilities"}
```

Key details:
- NO pooling (no CLS, no mean). Classifier operates on each token.
- Loss: `cross_entropy` with `ignore_index=-100` for [CLS], [SEP], padding tokens.
- `sanitize()` method: same weight remapping as SequenceClassification (encoder weights identical).

### 2. `mlx_trainer/token_classification_collator.py` (NEW)

**Aligns BIO labels to subword tokens.**

```
TokenClassificationCollator(tokenizer, max_length, label2id):
  __call__(features):
    # features = {"tokens": List[List[str]], "labels": List[List[str]]}
    # OR       = {"text": List[str], "bio_tags": List[List[str]]}

    tokenizer_output = tokenizer(tokens, padding=True, truncation=True,
                                  is_split_into_words=True)

    # Align labels:
    word_ids = tokenizer_output.word_ids()
    for each word_id:
      if word_id is None:          → -100  ([CLS], [SEP], pad)
      if first subword of word:    → B-XXX label
      if subsequent subword:       → I-XXX label (continuation)

    return {"input_ids", "attention_mask", "labels"}
```

Key details:
- `is_split_into_words=True` — tokenizer receives pre-tokenized words.
- `word_ids()` — maps each subword back to its original word index.
- First subword of a word gets the original label; subsequent subwords get the I- continuation.
- Special tokens ([CLS], [SEP], [PAD]) → `-100` (ignored in loss).

### 3. `mlx_trainer/token_classification_trainer.py` (NEW)

**Trainer subclass with entity-level evaluation.**

```
TokenClassificationTrainer(Trainer):
  - Override collator: use TokenClassificationCollator
  - Override evaluate():
      - Collect all predictions + labels (skip -100)
      - Convert back to entity spans using label2id / id2label
      - Compute entity-level precision, recall, F1 using seqeval
      - Also compute token-level accuracy
  - Override loss_fn():
      - Add class weights to handle O vs entity imbalance
      - weight[O] = 0.1-0.3 (downweight the majority class)
```

Key details:
- Inherits from `Trainer` for all training loop logic (gradient accumulation, checkpointing, logging).
- Only overrides `evaluate()` and the loss function.
- Uses `seqeval` library for entity-level metrics (P/R/F1 per entity type).
- Class weights: configurable, default `[1.0] * num_labels` with `O` downweighted to `0.2`.

---

## Files to Modify

### 4. `mlx_trainer/load.py` (MODIFY)

Add `load_token_classification()` function alongside existing `load()`.

```python
def load_token_classification(repo_id, model_config=None, train=False, dtype=mx.float16):
    # Same as load() but instantiates ModelForTokenClassification
    # instead of ModelForSequenceClassification
```

### 5. `mlx_trainer/modernbert_config.py` (MODIFY)

Add token-classification config fields to `ModelArgs`:

```python
# Add these fields:
class_weights: Optional[List[float]] = None  # per-label loss weights
token_classification: bool = False            # flag for model selection
```

### 6. `mlx_trainer/__init__.py` (MODIFY)

Export new classes:

```python
from .token_classification_model import ModelForTokenClassification
from .token_classification_collator import TokenClassificationCollator
from .token_classification_trainer import TokenClassificationTrainer
from .load import load_token_classification
```

---

## Files to Create (Training Script)

### 7. `main_pii.py` (NEW)

End-to-end PII training script, analogous to existing `main.py`.

```
Pipeline:
  1. Load dataset (e.g., ai4privacy/pii-masking-openpii-1m)
  2. Parse BIO labels, build label2id / id2label mapping
  3. Split train/eval
  4. Load model via load_token_classification()
  5. Create TokenClassificationTrainer
  6. Train
```

### 8. `inference_pii.py` (NEW)

Inference demo for PII detection, analogous to `inference.py`.

```
Pipeline:
  1. Load trained model
  2. Tokenize input
  3. Run inference
  4. Convert token predictions → entity spans
  5. Print highlighted PII entities
```

---

## Label Mapping (BIO Format)

```python
LABELS = [
    "O",              # Outside (no PII)
    "B-PER",          # Person name - beginning
    "I-PER",          # Person name - continuation
    "B-EMAIL",        # Email address
    "I-EMAIL",
    "B-PHONE",        # Phone number
    "I-PHONE",
    "B-IBAN",         # Bank account
    "I-IBAN",
    "B-CREDIT_CARD",  # Credit card
    "I-CREDIT_CARD",
    "B-IP_ADDRESS",   # IP address
    "I-IP_ADDRESS",
    "B-ADDRESS",      # Physical address
    "I-ADDRESS",
    "B-SSN",          # Social security / ID number
    "I-SSN",
    "B-USERNAME",     # Username / handle
    "I-USERNAME",
]

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}
```

---

## Dependency Addition

```toml
# pyproject.toml — add:
seqeval = ">=1.2.0"  # entity-level NER metrics
```

---

## Execution Order

1. **Step 1**: Add `seqeval` to `pyproject.toml`
2. **Step 2**: Create `token_classification_model.py`
3. **Step 3**: Create `token_classification_collator.py`
4. **Step 4**: Create `token_classification_trainer.py`
5. **Step 5**: Modify `load.py` — add `load_token_classification()`
6. **Step 6**: Modify `modernbert_config.py` — add fields
7. **Step 7**: Update `__init__.py` — exports
8. **Step 8**: Create `main_pii.py` — training script
9. **Step 9**: Create `inference_pii.py` — inference demo
10. **Step 10**: Test with a small subset before full training

---

## Risk Areas

| Risk | Impact | Mitigation |
|------|--------|------------|
| Subword label alignment bugs | Wrong labels → model learns garbage | Unit test collator with known examples |
| Class imbalance (95% O) | Model predicts all O | Weighted loss, seqeval for entity-level F1 |
| Memory (PII datasets are large) | OOM | Gradient checkpointing already in lib, use it |
| seqeval dependency | Extra dep | Lightweight, pure Python, no native deps |
