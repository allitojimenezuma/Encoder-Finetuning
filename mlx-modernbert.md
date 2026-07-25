---
name: "mlx-modernbert"
description: "A pure-MLX fine-tuning framework for ModernBERT on Apple Silicon, built from scratch to train a ~550M parameter encoder for text classification on a MacBook Pro."
tags: ["AI", "MLX", "Apple Silicon", "NLP", "Python"]
image: '../../../../public/static/projects/images/placeholder.png'
<!-- IMAGE RECOMMENDATION: replace placeholder with a screenshot of the training log
     showing the RAM / Loss / Speed lines (e.g. "Step 120 | Loss: 0.3669 | RAM - Active:
     5.9GB | Peak: 9.9GB"). A second good image: the final Eval line
     "Eval - loss: 0.0367 | acc: 0.9881 | f1: 0.9881". Keep both dark-terminal shots
     for visual consistency. -->
link: "https://github.com/allitojimenezuma/Encoder-Finetuning"
metric: "F1 0.988 on prompt-injection detection"
startDate: 2026-07-01
endDate: 2026-07-31
featured: true
---

## At a Glance

> I built **mlx-modernbert**, a from-scratch fine-tuning framework for the **ModernBERT** encoder that runs natively on Apple Silicon with **MLX** — no PyTorch, no TensorFlow. It trains a **~550M parameter** model on a **MacBook Pro M5 Pro (48GB)** using the Metal GPU and Neural Engine, reaching **F1 0.988** on a prompt-injection detection task.

- **Role:** Sole author — model implementation, training engine, evaluation, packaging.
- **Stack:** **Python 3.13**, **MLX**, **ModernBERT-large**, HuggingFace Datasets/Tokenizers, **seqeval**, scikit-learn.
- **Hardware:** MacBook Pro **M5 Pro**, 48GB unified memory, Apple Neural Engine / Metal.
- **Key result:** **F1 0.988**, **precision 0.988**, **recall 0.988** on 18,546 curated samples.
- **Repo:** [github.com/allitojimenezuma/Encoder-Finetuning](https://github.com/allitojimenezuma/Encoder-Finetuning)

<!-- IMAGE RECOMMENDATION: place the training-log screenshot right after this block
     so a recruiter sees real numbers within the first scroll. -->

## The Problem & Objective

I wanted to fine-tune a **ModernBERT-large** encoder on my MacBook Pro for text classification, taking advantage of the **M5 Pro's Neural Engine** and Metal GPU. I could not find a library that was efficient on Apple Silicon, reported good metrics, and supported features I needed like **resuming from a checkpoint** with optimizer state. So I built my own training framework from scratch on **MLX**, Apple's native array library, with no PyTorch dependency.

## System Architecture

Raw text enters a curated **HuggingFace Dataset**, flows through a **collator** that tokenizes and aligns labels, then a compiled training loop runs forward, backward, and optimizer update steps on Metal. The **Trainer** logs loss, learning rate, speed, and live RAM, then saves rotating checkpoints as **safetensors**. Evaluation runs a separate compiled function and computes classification metrics with scikit-learn.

- **Core Stack:** Python 3.13, **MLX**, ModernBERT-large, HuggingFace Datasets/Tokenizers.
- **Evaluation:** scikit-learn (sequence classification), **seqeval** (token classification, entity-level).
- **Key Architectural Choice:** Pure **MLX** instead of PyTorch so the whole pipeline runs on Metal with no translation layer, and compiled graphs via **`mx.compile`**.

## The Model: ModernBERT, Reimplemented

I reimplemented **ModernBERT-large** from scratch as `mlx.nn.Module` layers rather than pulling a model from a zoo. The architecture follows the original: **token embeddings** with **LayerNorm** and dropout, **RoPE** positional encoding, a hybrid attention that mixes **local sliding-window** attention (window 128) with **global** attention every 3 layers, **GLU** feed-forward blocks, and **pre-norm** residual connections. Pooling for classification is configurable to **CLS token** or **mean** pooling.

The model reads its shape from a **`ModelArgs`** dataclass built out of the HuggingFace `config.json`. A `from_dict` constructor copies only the known fields and ignores the rest, so new or vendor-specific keys never break loading. The config carries the full architecture, the classification head, the label maps, and task flags such as `is_regression` or `token_classification`.

### Loading Checkpoints into the Native Model

The loader pulls `answerdotai/ModernBERT-large` from the Hub, reads `config.json`, and instantiates **`ModelForSequenceClassification`**. Weights are cast to **fp16** by default. A **`sanitize`** method remaps HuggingFace parameter names to my module's names — for example `bert.` becomes `model.`, and stale keys like `position_ids` and `decoder.bias` are dropped. Any weight present in the model but missing from the checkpoint is initialized: **zero** for biases, **one** for norms, and a normal draw for the rest. The log confirms this with `Initialized 2 missing weights` — exactly the fresh classifier head.

<!-- IMAGE RECOMMENDATION: a cropped screenshot of the load lines
     "[load] Initialized 2 missing weights (mlx.core.float16)" and
     "[load] answerdotai/ModernBERT-large @ mlx.core.float16" shows the fp16
     loading and head init clearly. -->

## The Trainer: Memory and Compute

The **`Trainer`** is the core of the framework. I split the work into three **`mx.compile`**-compiled functions so Metal builds a single graph for each phase: **`step_fn`** computes loss and gradients, **`update_fn`** scales, clips, and applies the optimizer step, and **`eval_fn`** runs the forward pass for evaluation. Loss is computed in **fp32** for numerical stability even though parameters live in **fp16**.

### How I Optimized RAM

On a 48GB machine a **~550M parameter** model can saturate unified memory fast, so I layered several defenses. **fp16** halves the parameter footprint with no measurable accuracy loss. **Gradient checkpointing** wraps each of the 28 encoder layers with **`mx.checkpoint`**, so activations are dropped on the forward pass and recomputed only during the backward pass — trading compute for memory. After every optimizer step I call **`mx.clear_cache`** to release intermediate buffers, and after each logging window I call **`mx.reset_peak_memory`** so the reported peak stays per-window and comparable.

During evaluation I go further: a **`gc.collect`** plus **`mx.clear_cache`** before the loop stops training memory from bleeding into eval, and inside the loop I `del` the local tensors and clear the cache after every batch so the long-sequence eval never climbs into OOM. The reward shows up directly in the live log: **Active 5.9GB**, **Cache 0.8GB**, and the **Peak** falling from **14.1GB** to **9.9GB** once the cache is flushed.

<!-- IMAGE RECOMMENDATION: screenshot of three consecutive log blocks showing the
     Peak dropping from 14.1GB to 9.9GB — the clearest visual proof of the RAM
     optimizations working. -->

### How I Implemented Metrics

For sequence classification the **`evaluate`** method runs the compiled eval function, forces **`mx.eval`** so the graph materializes, then moves predictions and labels to **NumPy** so they stop stressing the MLX cache. From there scikit-learn's **`precision_recall_fscore_support`** with a `weighted` average produces **precision**, **recall**, and **F1**, alongside raw **accuracy** and **eval_loss**. Token classification reuses the same loop but reconstructs BIO sequences per example and hands them to **seqeval** in **strict** mode for entity-level **P/R/F1** per entity type.

The trainer also surfaces **training health**, not just final metrics: each logging window prints **loss**, the scheduled **learning rate**, **steps/s**, and the three live memory counters. That made it possible to watch the model converge and the memory behave in real time.

### Checkpoints and Resume

Checkpoints are written by **`_save_checkpoint`** to a `checkpoint-{step}` directory that holds **`model.safetensors`**, **`optimizer.safetensors`**, the tokenizer, **`config.json`**, and **`metrics.json`**. A **`_rotate_checkpoints`** routine keeps only the `save_total_limit` most recent checkpoints, so disk does not fill up over a long run. **Resume from checkpoint** restores both the model weights and the full optimizer state, so training continues with AdamW momentum intact instead of restarting cold — the feature I originally missed in other libraries.

## Engineering Trade-Offs

### Challenge: Memory Pressure on 48GB Unified Memory

- **The Bottleneck:** A **~550M parameter** encoder at **fp32** plus optimizer state and activations can exceed available unified memory on long sequences.
- **The Trade-off:** Keep everything in **fp32** for safety, or accept **fp16** plus recomputation to fit comfortably in 48GB.
- **The Solution:** **fp16** weights and a **fp32** loss, **gradient checkpointing** on all 28 layers, and explicit **`mx.clear_cache`** after each step and each eval batch. Peak settled around **9.9GB**.

### Challenge: Subword-to-Label Alignment for Token Classification

- **The Bottleneck:** The tokenizer splits words into subwords, but BIO labels are word-level, so naive alignment corrupts the loss target.
- **The Trade-off:** Label only the first subword and ignore the rest, or propagate labels through every subword with correct BIO continuation.
- **The Solution:** The **`TokenClassificationCollator`** uses `word_ids()` to map each subword to its source word, sets special tokens to **`-100`** (ignored), and converts later subwords of a `B-` entity into their `I-` continuation.

## Results & Impact

I trained a **prompt-injection guardrail** on 18,546 curated samples (10,343 SAFE, 8,203 INJECTION) drawn from `allenai/wildjailbreak`, `deepset/prompt-injections`, and `HuggingFaceH4/no_robots`, after stratified split and deduplication.

- **F1: 0.988** (precision 0.988, recall 0.988) on the held-out eval split.
- **Accuracy: 0.9881** with **eval_loss 0.0367** — near-ceiling on a non-trivial adversarial set.
- **Memory:** Stable **Active 5.9GB**, **Cache 0.8GB**, peak driven down to **9.9GB** from **14.1GB**.
- **Speed:** **8.9–10.4 steps/s** on MacBook Pro **M5 Pro**, batch 4 with gradient accumulation 8.
- **Zero PyTorch:** The entire stack runs on **MLX** and Metal, with only tokenizers and data utilities pulled from HuggingFace.

<!-- IMAGE RECOMMENDATION: final screenshot of the Eval line
     "Eval - loss: 0.0367 | acc: 0.9881 | f1: 0.9881" to close the results section
     with the headline number visible. -->

## Token Classification: PII Detection

The same framework extends to **token classification** through **`TokenClassificationTrainer`**, which swaps in the BIO-aware collator and a **weighted cross-entropy** loss. The PII schema covers entities like **PER**, **EMAIL**, **PHONE**, **IBAN**, and **SSN**, and the usual ~95% "O" class is down-weighted to **0.2** against **1.0** for entity classes. An end-to-end script lives at `examples/pii/train_pii.py` with matching inference in `examples/pii/inference_pii.py`.

I am currently working on the PII evaluation metrics and will update this page with entity-level **precision**, **recall**, and **F1** numbers once the runs are complete.

## Future Improvements

- Add **bf16** or mixed-precision options for users who want a speed/memory dial beyond **fp16**.
- Add learning-rate and loss **CSV/TensorBoard-style logging** so runs can be compared visually after the fact.
