# BUGS: Token Classification Review

Found during code review on 2025-07-16. Two bugs, two warnings.

---

## 🔴 Bug 1: Redundant Loss Computation in Training Step

**File:** `mlx_trainer/token_classification_trainer.py`, `_recompile_weighted_loss()` method

```python
def weighted_loss_fn(model, batch):
    out = model(**batch)       # ← batch includes "labels", model computes its own loss
    logits = out["logits"]     # ← but only logits are used; model's loss is discarded
    labels = batch["labels"]
    # ... computes weighted CE from logits
```

**Problem:** `model(**batch)` triggers the model's internal loss computation (lines 92-106 of `token_classification_model.py`), which does a full masked cross-entropy. That result is thrown away — the trainer recomputes loss with class weights from raw logits. Wasted compute on every training step.

**Fix:** Strip `labels` from batch before passing to model:

```python
def weighted_loss_fn(model, batch):
    model_input = {k: v for k, v in batch.items() if k != "labels"}
    out = model(**model_input)  # skips internal loss computation
    logits = out["logits"]
    labels = batch["labels"]
    # ... rest stays the same
```

**Impact:** ~5% training speedup. Not catastrophic if left as-is — the model's internal loss is a single masked cross-entropy, not expensive.

---

## ⚠️ Warning 1: Eval Loss vs Training Loss Scale Mismatch

**File:** `mlx_trainer/token_classification_trainer.py`

- `step_fn` (training) uses **weighted** CE (class weights applied)
- `eval_fn` (evaluation) uses **unweighted** CE (model's internal loss)

```python
# step_fn — weighted
weighted_ce = ce * flat_weights  # O class downweighted to 0.2

# eval_fn — unweighted
loss = mx.mean(out["loss"])  # model's internal CE, no class weights
```

**Result:** Training loss and eval loss are on different scales. During training logs:

```
Step 100 | Loss: 0.034    ← weighted (smaller, O is downweighted)
Eval — loss: 0.182        ← unweighted (larger, O contributes fully)
```

**Not a bug** — monitoring unweighted eval loss is correct for interpretability. But it can confuse someone reading the logs. Document it or add a comment in the code.

Also: `mx.mean(out["loss"])` is redundant. `out["loss"]` is already a scalar (mean over valid tokens computed in the model). No harm, just noise.

---

## ⚠️ Warning 2: Per-Example Tokenization is Slow

**File:** `mlx_trainer/token_classification_collator.py`

```python
for words, labels in zip(batch_words, batch_labels):
    encoding = self.tokenizer(words, is_split_into_words=True, ...)
```

**Problem:** Tokenizes one example at a time in a Python loop. Correct approach (need `word_ids()` per example), but 5-10x slower than batch tokenization. For a 1M-sample dataset, this becomes the training bottleneck.

**Fix (deferred):** Pre-tokenize the dataset once before training:

```python
# In main_pii.py, before creating trainer:
def tokenize_example(example):
    encoding = tokenizer(example["tokens"], is_split_into_words=True,
                         truncation=True, max_length=512)
    word_ids = encoding.word_ids()
    aligned_labels = align_labels(example["labels"], word_ids)
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": aligned_labels,
    }

tokenized_train = train_dataset.map(tokenize_example, remove_columns=train_dataset.column_names)
```

Then the collator only does padding. Not blocking for initial training, but important for large datasets.

---

## Summary

| #   | Severity   | File                               | Issue                                       |
| --- | ---------- | ---------------------------------- | ------------------------------------------- |
| 1   | 🔴 Bug     | `token_classification_trainer.py`  | Redundant model loss computation every step |
| 2   | ⚠️ Warning | `token_classification_trainer.py`  | Train/eval loss on different scales         |
| 3   | ⚠️ Warning | `token_classification_collator.py` | Per-example tokenization is slow at scale   |
