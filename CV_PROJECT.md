# CV Project Sheet: mlx-modernbert

---

## Project Name

**mlx-modernbert** — Pure MLX Training Framework for ModernBERT on Apple Silicon

## Tagline

Custom, from-scratch MLX implementation of the ModernBERT encoder with a full fine-tuning pipeline (sequence & token classification), zero PyTorch/TensorFlow dependency, running natively on Apple Metal GPU.

## Links

- **Repository:** https://github.com/allitojimenezuma/Encoder-Finetuning
- **PyPI package:** `mlx-modernbert` (v0.1.0)
- **License:** MIT

---

## One-Line Summary

Built a complete MLX-native training framework that reimplements the ModernBERT architecture from scratch and provides production-ready trainers for text classification (e.g., prompt-injection guardrails) and token classification (e.g., PII detection) — all on Apple Silicon Metal.

---

## Technical Highlights

| Area | Detail |
|---|---|
| **Architecture** | Self-contained ModernBERT-large implementation (~1,700 LOC): embeddings with LayerNorm + dropout, RoPE positional encoding, hybrid local (sliding window) + global attention, GLU feed-forward blocks, pre-norm residual connections, mean/CLS pooling. |
| **Training Engine** | Custom trainer with `mx.compile`-compiled forward/backward/update steps, gradient accumulation, mixed-precision fp16 training, gradient checkpointing, cosine/linear LR scheduling with warmup, and checkpoint rotation. |
| **Token Classification** | Full BIO-format support: subword-to-label alignment via `word_ids()`, weighted cross-entropy for class imbalance (down-weighting "O" class), entity-level P/R/F1 evaluation via `seqeval`, and per-entity-type metrics. |
| **Sequence Classification** | Prompt-injection detection (SAFE/INJECTION), binary/multi-class/multi-label classification with configurable pooling (CLS token or mean). |
| **Memory Optimization** | fp16 (50% VRAM reduction), gradient checkpointing on encoder layers, compiled evaluation functions, aggressive cache clearing between eval batches to avoid OOM on long sequences. |
| **Packaging** | Published as a pip-installable package (`mlx-modernbert`) with `pyproject.toml`, hatchling build, and full dependency chain (mlx ≥0.24, transformers, datasets, seqeval, scikit-learn). |
| **Model Loading** | Automatic loading of HuggingFace `answerdotai/ModernBERT-large` checkpoints into the native MLX implementation with key sanitization (HF → MLX parameter mapping). |

---

## Key Skills Demonstrated

### Machine Learning / Deep Learning
- Modern transformer architecture implementation from scratch (not using a model zoo)
- Attention mechanisms: RoPE, local sliding window, global attention, scaled dot-product
- Training techniques: gradient accumulation, gradient checkpointing, mixed-precision (fp16), early stopping
- Loss functions: weighted cross-entropy for severe class imbalance, binary/multi-class cross-entropy, MSE regression
- Evaluation: entity-level precision/recall/F1 with seqeval, token accuracy, sequence-level metrics

### Apple MLX Framework
- Custom model implementation using `mlx.nn.Module`, `mlx.core`, `mlx.optimizers`
- Compiled training/evaluation graphs via `mx.compile` for Metal GPU acceleration
- Memory management: `mx.get_active_memory()`, `mx.get_peak_memory()`, `mx.clear_cache()`
- Safetensors serialization/deserialization for checkpoints

### Software Engineering
- Clean Python packaging (pyproject.toml, hatchling, uv)
- Type hints throughout (dataclasses, Optional, Dict, Literal)
- Modular architecture: model, config, collator, trainer separated into distinct modules
- Test coverage: unit tests for collator, training, token classification
- CLI inference scripts for both classification tasks

### MLOps / ML Engineering
- HuggingFace Hub integration (model download, tokenizer handling)
- Checkpoint management with automatic rotation
- Resume-from-checkpoint with optimizer state restoration
- Training metrics logging (loss, LR, memory usage, speed)

---

## Project Scope

- **1,701** lines of Python across 11 modules
- **2** classification heads (sequence + token)
- **2** example applications (prompt-injection guardrail + PII detector)
- **3** test files covering collator logic, training, and token classification
- Published as a reusable pip package

---

## Use Cases / Applications

1. **Prompt Injection Detection** — Binary classifier to detect malicious prompt-injection attacks against LLMs (SAFE vs INJECTION labels).
2. **PII Detection / NER** — Token-level entity recognition to identify personal information (names, emails, phone numbers) using BIO-format labeling.
3. **General Text Classification** — Any sequence-level classification task on Apple Silicon hardware.

---

## Technical Challenges Solved

1. **Subword-label alignment in token classification** — The tokenizer splits words into subwords but labels are word-level. Solved by implementing `word_ids()`-based alignment that propagates BIO labels to each subword token while correctly handling special tokens (set to -100 / ignored).

2. **Class imbalance in PII detection** — ~95% of tokens are "O" (non-entity). Solved with per-class weighted cross-entropy loss where "O" is down-weighted to 0.2× and entity classes keep 1.0× weight.

3. **Memory pressure on Apple Silicon** — ModernBERT-large (550M params) at fp32 exhausts unified memory. Solved with fp16 training, gradient checkpointing on encoder layers, compiled eval functions, and explicit cache management between evaluation batches.

4. **Hybrid attention mechanism** — ModernBERT uses local sliding-window attention on most layers and global attention every N layers. Implemented by building separate additive masks (global and windowed) and routing them per layer.

---

## Example: Training a PII Detector

```python
from mlx_trainer import TokenClassificationTrainer, TrainingArgs
from mlx_trainer.load import load_token_classification

model, tokenizer = load_token_classification(
    "answerdotai/ModernBERT-large",
    train=True,
    num_labels=6,
    id2label={0: "O", 1: "B-PER", 2: "I-PER", 3: "B-EMAIL", 4: "I-EMAIL", 5: "B-PHONE"},
    label2id={v: k for k, v in {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-EMAIL", 4: "I-EMAIL", 5: "B-PHONE"}.items()},
)

args = TrainingArgs(
    output_dir="outputs/pii_model",
    num_train_epochs=5,
    batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    grad_checkpoint=True,
)

trainer = TokenClassificationTrainer(
    model=model, tokenizer=tokenizer, training_args=args,
    train_dataset=train_dataset,
    id2label=id2label, label2id=label2id,
    class_weights=[0.2, 1.0, 1.0, 1.0, 1.0, 1.0],  # downweight "O"
)
trainer.train()
```

---

## Tech Stack

`Python 3.13` · `MLX (Apple)` · `ModernBERT` · `HuggingFace Transformers` · `HuggingFace Datasets` · `seqeval` · `scikit-learn` · `NumPy` · `Apple Metal GPU`
