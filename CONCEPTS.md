# mlx-modernbert — All Technical Concepts Explained

Every concept that appears in the codebase, explained from the ground up.

---

## 1. Gradient Checkpointing (memory vs compute trade-off)

**What it is:** A technique to reduce RAM by discarding intermediate activations during the forward pass and recomputing them only when needed during the backward pass.

**How it works normally:** During forward pass, layer 0 produces an output. That output is kept in memory because the backward pass needs it to compute gradients for layer 0's weights. Layer 1 takes it and produces its own output, also kept. For 28 layers, you store 28 activation tensors simultaneously.

**With gradient checkpointing:** Layer 0's output is produced, then **deleted from memory**. Layer 1 runs on its own, storing its output. During backward pass, to compute layer 0's gradients, you **re-run layer 0's forward pass** (cheap compute) to regenerate the activation, then compute the gradient. You do this layer by layer, so at any point only ~1-2 layers' activations exist in memory.

**In this code:** `_apply_grad_checkpointing` wraps each of the 28 encoder layers with `mx.checkpoint`. MLX does this under the hood — the wrapper tells the framework "don't cache my output; recompute it if the backward pass needs it." The result: **Active memory drops from 14.1GB to 5.9GB** because activations are no longer stored across all layers simultaneously. The cost is a ~20-30% slower training (each layer's forward is run twice).

---

## 2. mx.compile (Metal graph compilation)

**What it is:** MLX's equivalent of PyTorch's `torch.compile`. You declare a function, list its **input** and **output** state tensors, and MLX builds an optimized Metal GPU kernel the first time the function runs. Subsequent calls skip all Python overhead and run directly on the GPU.

**Why it matters:** Without compilation, every forward/backward step goes through Python interpretation, MLX dispatch, and Metal API calls. With compilation, the entire graph (forward → loss → gradients) is baked into one GPU kernel. This is where the 8-10 steps/s speed comes from.

**In the code:** Three compiled functions exist:
- `step_fn` — input: model state + RNG state; output: same; computes loss and gradients.
- `update_fn` — input: model + optimizer + RNG state; output: same; applies the optimizer step.
- `eval_fn` — input: model state; output: model state; runs forward pass for evaluation.

The `inputs`/`outputs` tell MLX which tensors change. If the same function is called with the same shapes, MLX reuses the compiled kernel. If shapes change (different batch size), it recompiles.

---

## 3. Gradient Accumulation

**What it is:** A way to simulate a larger batch size without using more memory per step.

**The problem:** Your M5 Pro has 48GB. A batch of 4 uses ~6GB. A batch of 32 would use ~48GB — no room for gradients, optimizer state, or activations. But larger batches give smoother gradients and better convergence.

**The solution:** Run 8 micro-batches of size 4 (total 32 samples), accumulate the gradients by summing them, then do one optimizer step. Mathematically equivalent to one batch of 32, but memory stays at the level of a batch of 4.

**In the code:**
```
accum = None
for raw_batch in self._batches(self.train_dataset, self.args.batch_size):
    loss, grads = self.step_fn(batch)
    if accum is None:
        accum = grads
    else:
        accum = tree_map(lambda a, b: a + b, accum, grads)

    if n_steps % accum_steps == 0:
        self.update_fn(accum, scale)   # scale = 1/accum_steps
```

The `scale` divides the accumulated gradients by the number of micro-batches (8), so the final update is an **average**, not a sum. Without scaling, gradients would be 8x too large.

---

## 4. fp16 (Half-Precision Floating Point)

**What it is:** Storing model weights and activations in 16-bit floats instead of 32-bit. Each parameter goes from 4 bytes to 2 bytes. For a 550M parameter model, that's ~1.1GB vs ~2.2GB for weights alone.

**Why it's safe:** The model's parameters are small numbers (typically -0.02 to 0.02 after initialization). fp16 has enough precision to represent these. The only place where 32-bit matters is during loss computation, where many small numbers accumulate — using fp32 there prevents numerical instability.

**In the code:** `model.set_dtype(dtype)` converts all weights to `mx.float16`. But `loss_fn` casts the result to `mx.float32`:
```python
def loss_fn(model, batch):
    out = model(**batch)
    return mx.mean(out["loss"]).astype(mx.float32)
```

This is the standard mixed-precision pattern: **forward/backward in fp16, loss and gradient accumulation in fp32**.

---

## 5. value_and_grad

**What it is:** MLX's function that takes a model and a loss function, and returns a new function that computes **both** the loss value **and** the gradients of the loss with respect to all model parameters.

**Why it exists:** Manually computing gradients through 28 transformer layers is impossible. `value_and_grad` automates it using **automatic differentiation** (backpropagation). You give it the loss function, it traces every operation, and produces a gradient tensor for every weight in the model.

**In the code:**
```python
grad_fn = nn.value_and_grad(self.model, loss_fn)
loss, grads = grad_fn(self.model, batch)
```

`loss` is a scalar (the training loss). `grads` is a tree structure matching the model's parameter tree, where each leaf is the gradient tensor for that parameter. These gradients are then accumulated (gradient accumulation) and fed to `update_fn`.

---

## 6. AdamW Optimizer

**What it is:** The optimizer that adjusts model weights based on gradients. It's a variant of Adam (Adaptive Moment Estimation) with decoupled weight decay.

**How Adam works:** For each parameter, Adam tracks two running averages: the **mean** (first moment) and the **variance** (second moment) of the gradient. It divides the gradient by the square root of the variance, which automatically adapts the learning rate per-parameter: parameters with noisy gradients get smaller updates, parameters with consistent gradients get larger ones.

**What "W" adds:** Weight decay penalizes large weights by subtracting a fraction of the weight from itself at each step (`weight *= 1 - lr * weight_decay`). This is separate from the gradient-based update, hence "decoupled."

**In the code:** `opt.AdamW(learning_rate=schedule, weight_decay=0.1)` initializes with `weight_decay=0.1` (10% per step). The `learning_rate` is not a fixed number but a schedule function that changes the LR over time.

---

## 7. Learning Rate Schedule (cosine decay + warmup)

**What it is:** Instead of using a fixed learning rate, you vary it over training. Two phases:

**Warmup:** For the first 10% of steps (`warmup_ratio=0.1`), the LR ramps linearly from 0 to the target (e.g., 1e-5). This prevents large gradients at the start when the model is random and the loss landscape is steep.

**Cosine decay:** After warmup, the LR follows a cosine curve from the target down to near-zero. This lets the model make large progress early and fine-tune near the end.

**Why cosine vs linear:** Cosine decays slowly at first (fast learning) and quickly at the end (fine-grained convergence). Linear decays at a constant rate. Cosine typically converges better because early steps get more learning.

**In the code:**
```python
warmup_fn = opt.linear_schedule(0.0, args.learning_rate, warmup_steps)
schedule_fn = opt.cosine_decay(args.learning_rate, decay_steps)
return opt.join_schedules([warmup_fn, schedule_fn], [warmup_steps])
```

`join_schedules` chains two functions: warmup runs for `warmup_steps`, then cosine takes over.

---

## 8. Gradient Clipping

**What it is:** A safety mechanism that prevents a single batch from causing a huge weight update that destabilizes training.

**The problem:** If one batch has unusual examples, the gradient can be very large. Multiplying the current weights by this gradient can throw the model into a random state it cannot recover from.

**The solution:** Compute the **global norm** (total magnitude) of all gradient tensors. If it exceeds a threshold (`max_grad_norm=1.0`), scale all gradients down so the total norm equals the threshold. This preserves the gradient's direction but limits its magnitude.

**In the code:**
```python
clipped, _ = opt.clip_grad_norm(scaled, self.args.max_grad_norm)
self.optimizer.update(self.model, clipped)
```

---

## 9. mx.clear_cache / mx.reset_peak_memory / gc.collect

These three tools manage memory on Apple Silicon:

**`mx.clear_cache()`** — tells MLX to release all cached GPU memory (intermediate tensors from previous steps). Without this, old tensors pile up and you get OOM. Called after every optimizer step and after every eval batch.

**`mx.reset_peak_memory()`** — resets the "peak memory" counter. The trainer logs peak memory per logging window, not cumulative. This makes the numbers comparable across windows.

**`gc.collect()`** — Python's garbage collector. MLX tensors are backed by Python objects. Sometimes Python's reference counting doesn't free them immediately (circular references, delayed cleanup). `gc.collect()` forces immediate cleanup. Called before evaluation to stop training memory from bleeding into eval.

---

## 10. RoPE (Rotary Position Embeddings)

**What it is:** A way to encode token positions into the attention mechanism without adding position vectors. Instead of learning a position embedding and adding it to the token, RoPE **rotates** the query and key vectors by an angle proportional to their position.

**How it works:** Imagine each head's query/key vector as a point in 2D space. Position 0 is unrotated. Position 1 is rotated by angle θ. Position 2 by 2θ. When you compute dot product (attention score) between query at position i and key at position j, the rotation difference is proportional to `(i-j)` — the relative distance. So the model learns relative positions without explicit position embeddings.

**In the code:** `nn.RoPE(dims=self.head_dim, base=rope_theta)` is applied to Q and K before computing attention. ModernBERT uses two thetas: `global_rope_theta=160000.0` for global attention layers and `local_rope_theta=10000.0` for local attention layers, because local layers have a smaller context window and need different frequency scaling.

---

## 11. Local vs Global Attention (Sliding Window)

**What it is:** A hybrid attention pattern that reduces compute cost. Most layers use **local attention** (each token only attends to neighbors within a window). Every N layers, **global attention** lets every token attend to every other token.

**Why it exists:** Full attention on 550M parameters is expensive: O(n²) in sequence length. But you still need some global information flow. The solution: most layers are cheap (local window of 128), and every 3 layers (`global_attn_every_n_layers=3`) you do full attention to let distant tokens communicate.

**In the code:** The mask building logic in `_update_attention_mask` creates two masks:
- **Global mask:** standard causal/padding mask — every token can see every other token.
- **Sliding mask:** a band matrix — each token can only see tokens within ±64 positions (half of `local_attention=128`).

Each encoder layer picks which mask to use based on its layer ID:
```python
if layer_id % config.global_attn_every_n_layers != 0:
    mask = sliding_window_mask  # local
else:
    mask = attention_mask        # global
```

---

## 12. GLU (Gated Linear Unit) Feed-Forward

**What it is:** A variant of the standard feed-forward (FFN) layer that uses a gating mechanism. Instead of one linear layer → activation → second linear layer, GLU splits the intermediate into two halves: one is the **content**, the other is the **gate**.

**How it works:**
```
x = Linear(hidden → 2 * intermediate)   # projects to double width
inp, gate = split(x)                      # first half = content, second half = gate
output = Linear(intermediate * act(inp) * gate)  # gate controls how much content passes
```

The gate is a learned mask that can suppress or amplify specific dimensions of the content. This is more expressive than a simple activation because the gate is learned per-dimension.

**In the code:**
```python
def __call__(self, hidden_states):
    x = self.Wi(hidden_states)         # [batch, seq, 2*intermediate]
    split = x.shape[-1] // 2
    inp, gate = x[..., :split], x[..., split:]
    return self.Wo(self.drop(self.act(inp) * gate))
```

`nn.GELU` is used as the activation on the content half. The gate half has no activation (it stays linear).

---

## 13. Pre-norm Residual Connections

**What it is:** A transformer layer design where LayerNorm is applied **before** the attention/FFN (not after). The residual connection adds the original input to the output.

**Why pre-norm:** In early transformers (GPT-2, original BERT), norm was after the sub-layer. Pre-norm was found to train more stably at depth because gradients flow through the residual path without passing through the norm, which can shrink them.

**In the code:**
```python
class ModernBertEncoderLayer:
    def __call__(self, hidden_states, ...):
        normed = self.attn_norm(hidden_states)       # norm BEFORE attention
        attn_out = self.attn(normed, ...)
        hidden_states = hidden_states + attn_out[0]  # residual ADD
        mlp_out = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + mlp_out       # residual ADD
```

The first layer (`layer_id=0`) uses `nn.Identity` as its attention norm — no normalization on the very first layer, which is a ModernBERT-specific choice.

---

## 14. CLS Pooling vs Mean Pooling

**What it is:** After the encoder produces a sequence of hidden states (one per token), you need a single vector to feed into the classification head. Two strategies:

**CLS pooling:** Take the hidden state of the `[CLS]` token (position 0). This is what BERT uses. The `[CLS]` token is trained to aggregate information from the entire sequence.

**Mean pooling:** Average all non-padding token hidden states. This is generally more robust for long sequences because it doesn't rely on a single token.

**In the code:**
```python
if self.config.classifier_pooling == "cls":
    pooled = last_hidden_state[:, 0]
else:
    pooled = _mean_pooling(last_hidden_state, attention_mask)
```

ModernBERT-large uses **mean pooling** by default (from `config.json`: `classifier_pooling: "mean"`). The mean pooling function masks out padding tokens before averaging.

---

## 15. Dropout

**What it is:** During training, randomly zero out a fraction of neurons. This prevents co-adaptation (neurons relying on specific other neurons) and acts as regularization.

**In the code:** ModernBERT has dropout at three points:
- Embedding dropout: `embedding_dropout=0.0` (disabled)
- Attention dropout: `attention_dropout=0.0` (disabled)
- MLP dropout: `mlp_dropout=0.0` (disabled)
- Classifier dropout: `classifier_dropout=0.0` (disabled)

All dropout is set to 0.0 in ModernBERT-large. The model relies on other regularization (weight decay, data augmentation from the dataset).

---

## 16. Weighted Cross-Entropy

**What it is:** A loss function where different classes contribute differently to the total loss. Used when classes are imbalanced.

**The problem:** In PII detection, ~95% of tokens are "O" (non-entity). If the model predicts "O" for everything, it gets 95% accuracy — but detects nothing. The loss treats all classes equally, so the model can ignore rare entity classes.

**The solution:** Multiply the loss for each token by a class weight. "O" gets weight 0.2, entity classes get weight 1.0. Now the model is penalized 5x more for missing an entity than for misclassifying an "O".

**In the code:**
```python
def weighted_loss_fn(model, batch):
    ce = nn.losses.cross_entropy(flat_logits, safe_labels)
    flat_weights = weights[safe_labels]        # look up weight for each label
    weighted_ce = ce * flat_weights            # scale the loss
    masked = mx.where(valid_mask.reshape(-1), weighted_ce, mx.zeros_like(weighted_ce))
    return mx.sum(masked) / valid_count
```

The `valid_mask` ensures tokens with label `-100` (padding/special) contribute zero loss.

---

## 17. -100 Label (Ignore Index)

**What it is:** A magic integer that tells the loss function "skip this token entirely." PyTorch and MLX both use `-100` as the default `ignore_index` for cross-entropy.

**Why it exists:** You don't want to train on padding tokens, `[CLS]`, `[SEP]`, or subword continuations. Setting their labels to `-100` means the loss function returns 0 for them, and gradients are never computed.

**In the code:** TokenClassificationCollator sets special tokens to `-100`:
```python
if word_id is None:
    aligned_labels.append(-100)
```

The weighted loss function also masks them:
```python
valid_mask = labels != -100
masked = mx.where(valid_mask.reshape(-1), weighted_ce, mx.zeros_like(weighted_ce))
```

---

## 18. Scaled Dot-Product Attention

**What it is:** The core attention formula. For each query, compute its similarity to all keys, scale, apply softmax, and use the resulting weights to aggregate values.

**Formula:** `Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V`

**Why scale by sqrt(d_k):** Without scaling, dot products grow with dimension. If d_k=64, dot products are ~64x larger than they should be, pushing softmax into saturation (all weights near 0 or 1), making gradients vanish.

**In the code:**
```python
scale = query.shape[-1] ** -0.5   # 1/sqrt(64) = 0.125
attn_output = mx.fast.scaled_dot_product_attention(
    query, key, value, scale=scale, mask=mask
)
```

MLX has a fused kernel (`mx.fast`) that computes this entire operation in one Metal GPU pass, which is much faster than doing it in separate steps.

---

## 19. word_ids()

**What it is:** A method from HuggingFace tokenizers that returns a list mapping each token back to the word it came from.

**Example:** For `["Alice", "lives", "in", "Paris"]` tokenized into `["[CLS]", "Al", "##ice", "lives", "in", "Par", "##is", "[SEP]"]`, `word_ids()` returns `[None, 0, 0, 1, 2, 3, 3, None]`. The `None` values are special tokens.

**Why it's critical:** Without this, you cannot align word-level labels to subword tokens. The collator uses it to implement the BIO alignment algorithm explained earlier.

---

## 20. Safetensors

**What it is:** A file format for storing tensor data. Similar to PyTorch's `.pt` files but safer (no arbitrary code execution) and faster to load.

**Why it exists:** Traditional pickle-based formats can execute arbitrary code when loaded. Safetensors is a flat binary format: just tensor shapes, dtypes, and raw data. It's also memory-mapped, so loading doesn't duplicate data.

**In the code:** Checkpoints are saved as `model.safetensors` and `optimizer.safetensors`:
```python
mx.save_safetensors(str(path / "model.safetensors"), weights)
```

---

## 21. Checkpoint Rotation

**What it is:** Automatically deleting old checkpoints to save disk space. Only the N most recent checkpoints are kept.

**Why it exists:** A training run might save hundreds of checkpoints (every 500 steps). ModernBERT-large at fp16 is ~1.1GB per checkpoint. 100 checkpoints = 110GB. Rotation keeps only the last 3.

**In the code:**
```python
def _rotate_checkpoints(self):
    limit = self.args.save_total_limit   # default 3
    checkpoints = sorted([d for d in self.output_dir.iterdir() ...], key=...)
    while len(checkpoints) > limit:
        old = checkpoints.pop(0)         # oldest first
        for f in old.iterdir():
            f.unlink()
        old.rmdir()
```

---

## 22. tree_flatten / tree_map

**What they are:** Utility functions for working with nested Python structures (dicts of dicts of tensors, etc.).

**`tree_flatten`**: Takes a nested structure and flattens it into a list of `(path, value)` pairs. Used to extract all parameters from the model into a flat dictionary for saving:
```python
weights = dict(tree_flatten(self.model.parameters()))
```

**`tree_map`**: Applies a function to every leaf in the tree, preserving structure. Used to scale all gradients uniformly:
```python
scaled = tree_map(lambda g: (g * scale).astype(mx.float32), accumulated_grads)
```

Both are standard MLX utilities because model parameters, optimizer states, and gradients are all nested tree structures, not flat arrays.

---

## 23. Weight Sanitization

**What it is:** Mapping parameter names from one format (HuggingFace checkpoint) to another (your MLX model).

**The problem:** HuggingFace checkpoints use names like `bert.encoder.layer.0.attn.Wqkv.weight`. Your MLX model uses `model.layers.0.attn.Wqkv.weight`. The names don't match. Loading fails.

**The solution:** A `sanitize` method that strips/renames prefixes:
```python
def sanitize(self, weights):
    sanitized = {}
    for k, v in weights.items():
        if "position_ids" in k:
            continue                        # not a learnable parameter
        if k.startswith("bert"):
            k = k.replace("bert.", "model.")
        sanitized[k] = v
    return sanitized
```

After sanitization, `_init_head_weights` fills in any weights that exist in your model but not in the checkpoint (like the freshly initialized classifier head).

---

## 24. Entity-Level Evaluation (seqeval)

**What it is:** A metric for NER/token classification that evaluates at the **entity span** level, not the token level.

**Token-level is misleading:** If you have `B-PER I-PER O O` and predict `B-PER O O O`, token accuracy is 75% (3/4 correct). But the entity "PER" was only partially detected — entity-level F1 would be lower.

**How seqeval works:** It reconstructs entity spans from BIO tags (e.g., `["Alice", "Smith"]` becomes entity `(PER, 0, 2)`), then compares predicted spans against gold spans. It computes precision (what fraction of predicted entities are correct), recall (what fraction of gold entities were found), and F1.

**In the code:**
```python
report_dict = seqeval_report(all_y_true, all_y_pred, mode="strict", output_dict=True)
micro = report_dict.get("micro avg", {})
entity_f1 = micro.get("f1-score", 0.0)
```

`mode="strict"` means an entity is only correct if both the type AND boundaries match exactly.
