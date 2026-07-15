# MLX Trainer Refactoring Plan

## 1. Project Overview

This repository contains a custom-built, Apple Silicon-optimized text classification trainer utilizing the `mlx` framework. The goal of this refactoring phase is to fix critical memory bottlenecks (due to lazy evaluation), prevent numerical instability (gradient clipping and underflow), and implement standard training features (metrics, optimizer state persistence, and early stopping).

The agent must strictly follow MLX best practices, particularly regarding graph compilation (`mx.compile`), lazy evaluation triggers (`mx.eval`), and tree operations (`mlx.utils.tree_map`).

---

## 2. Phase 1: Critical Bug Fixes (Priority: High)

### 2.1 Fix Lazy Evaluation in Gradient Accumulation

**File:** `trainer.py`
**Problem:** MLX builds computational graphs lazily. Accumulating gradients iteratively using `tree_map(lambda a, b: a + b, accum, grads)` without evaluating them causes the graph to grow linearly, resulting in massive RAM spikes and degraded performance.
**Action:**

- Inside `Trainer._train_epoch()`, immediately after accumulating gradients, force an evaluation of the `accum` tree.
- Add `mx.eval(accum)` right after the `accum = tree_map(...)` line to compute the graph at that step and free memory.

### 2.2 Implement Gradient Clipping

**File:** `trainer.py` (and verify `args.py`)
**Problem:** Exploding gradients can ruin training, especially in models like ModernBERT. The argument `max_grad_norm` exists in `TrainingArgs` but is unused.
**Action:**

- Import `clip_grad_norm` from `mlx.optimizers`.
- Inside `Trainer.update_fn` (or right before calling it), apply clipping to the accumulated gradients using `self.args.max_grad_norm`.
- _Agent Note:_ Ensure the clipping function is compatible with `mx.compile` if it's placed inside `update_fn`.

### 2.3 Mitigate FP16 Underflow Risk

**File:** `load.py` & `trainer.py`
**Problem:** Loading all weights and running the entire optimizer in strict FP16 can lead to gradient underflow and unstable LayerNorm operations.
**Action:**

- Ensure the loss computation inside `loss_fn` explicitly casts to `mx.float32` (this is already partially done, but ensure all intermediate reduction operations are FP32).
- Allow the optimizer state to be instantiated in FP32. If necessary, upcast gradients to FP32 before applying the optimizer update, then cast the updated weights back to FP16.

---

## 3. Phase 2: Memory & Performance Optimization (Priority: Medium)

### 3.1 Optimize Evaluation Loop Cache Clearing

**File:** `trainer.py`
**Problem:** Calling `mx.clear_cache()` inside the `for` loop in `Trainer.evaluate()` forces unnecessary synchronizations and destroys evaluation throughput.
**Action:**

- Remove `mx.clear_cache()` from the inside of the evaluation batch loop.
- Move `mx.clear_cache()` to execute only _after_ the entire evaluation loop has finished.

### 3.2 Update Step Compilation

**File:** `trainer.py`
**Problem:** Optimization logic is spread out.
**Action:**

- Ensure `update_fn` encompasses scaling, clipping, and the optimizer update step. This ensures maximum efficiency when executed as a single compiled graph on the Apple Silicon GPU.

---

## 4. Phase 3: Core Features & Functionality (Priority: High)

### 4.1 Save and Load Optimizer State

**File:** `trainer.py`
**Problem:** `_save_checkpoint` saves model weights but ignores `self.optimizer.state`. Resuming training from a checkpoint will reset momentum and Adam statistics.
**Action:**

- In `_save_checkpoint()`, extract the optimizer state using `tree_flatten(self.optimizer.state)`.
- Save it alongside the model weights as `optimizer.safetensors` using `mx.save_safetensors`.
- Add a `resume_from_checkpoint` parameter (optional string path) to `Trainer.__init__` or `TrainingArgs`. If provided, load the model weights, tokenizer, AND the optimizer state.

### 4.2 Implement Training Metrics (Accuracy, F1)

**File:** `trainer.py`
**Problem:** `evaluate()` only calculates `eval_loss`.
**Action:**

- Modify `evaluate()` to compute actual classification metrics.
- Extract `logits` from the model output.
- Compute predictions using `preds = mx.argmax(logits, axis=-1)`.
- Compare `preds` against `batch["labels"]`. Calculate at least Accuracy (total correct / total samples).
- _Agent Note:_ Return these metrics in the dictionary returned by `evaluate()`, print them to the console, and log them in `metrics.json`.

### 4.3 Add Early Stopping Mechanism

**File:** `args.py` & `trainer.py`
**Problem:** No way to stop training if validation loss degrades (overfitting).
**Action:**

- Add `early_stopping_patience: Optional[int] = None` to `TrainingArgs`.
- In `Trainer.train()`, track the best `eval_loss` and an integer counter for epochs without improvement.
- If the counter exceeds `early_stopping_patience`, print an early stopping message and `break` the training loop.

---

## 5. Phase 4: Code Design & Safety Improvements (Priority: Low)

### 5.1 Safe Pad Token Assignment

**File:** `collator.py`
**Problem:** Blindly falling back to `eos_token` for padding can confuse models with specific architectural expectations (like ModernBERT).
**Action:**

- Check `tokenizer.pad_token_id`. If `None`, check if the model config explicitly defines a pad token.
- If forced to use `eos_token`, issue a `logging.warning` so the user is aware of the fallback.

### 5.2 Robust Head Initialization

**File:** `load.py`
**Problem:** `_init_head_weights()` uses brittle substring matching (`"classifier"`, `"score"`, `"head"`).
**Action:**

- Instead of string matching, dynamically check which weights are missing from the loaded `.safetensors` checkpoint compared to the instantiated model parameters.
- Only initialize those specific missing tensors using the appropriate distribution (`normal` for weights, zeros for bias) rather than hardcoding layer name expectations.
