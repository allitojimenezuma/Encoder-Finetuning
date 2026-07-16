"""
🛡️ ModernBERT PII Detection — Fine-Tuning Script

End-to-end pipeline for token-level PII detection using BIO labeling.
Trains ModernBERT-large with weighted cross-entropy and entity-level eval.
"""
import argparse
import os

import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

# Parche para evitar el error de importación en la generación de model cards
try:
    import mlx_raclate.tuner.model_card_utils as model_card_utils
    model_card_utils.get_inference_code = lambda *args, **kwargs: "# Snippet omitido\n"
except ImportError:
    pass

os.environ.setdefault("HF_TOKEN", "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")

# ── Label schema ───────────────────────────────────────────────────
LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-EMAIL", "I-EMAIL",
    "B-PHONE", "I-PHONE",
    "B-IBAN", "I-IBAN",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-IP_ADDRESS", "I-IP_ADDRESS",
    "B-ADDRESS", "I-ADDRESS",
    "B-SSN", "I-SSN",
    "B-USERNAME", "I-USERNAME",
]

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}


# ── Dataset loading ────────────────────────────────────────────────

def load_pii_dataset(dataset_name: str, seed: int = 3407, test_size: float = 0.15):
    """Load a PII dataset and parse BIO labels.

    Supports datasets that expose:
      - `tokens` + `tags` (integer label ids), or
      - `tokens` + `labels` (BIO string labels), or
      - `text` + `bio_tags` (string-level BIO tags).

    Returns (train_dataset, eval_dataset) as HuggingFace Datasets with
    `tokens` (List[str]) + `labels` (List[str]) columns ready for
    TokenClassificationCollator.
    """
    print(f"\n[1/4] Loading PII dataset: {dataset_name}...")

    ds = load_dataset(dataset_name, split="train")

    # Inspect columns to determine format
    cols = set(ds.column_names)
    print(f"  Columns: {sorted(cols)}")

    rows = []

    if "tokens" in cols and "tags" in cols:
        # Format: tokens (List[str]), tags (List[int]) — needs label mapping
        print("  -> Detected tokens+tags format (integer labels)")
        for example in ds:
            tokens = example["tokens"]
            tag_ids = example["tags"]
            labels = [id2label.get(t, "O") for t in tag_ids]
            rows.append({"tokens": tokens, "labels": labels})

    elif "tokens" in cols and "labels" in cols:
        # Format: tokens (List[str]), labels (List[str]) — already BIO strings
        print("  -> Detected tokens+labels format (string labels)")
        for example in ds:
            rows.append({"tokens": example["tokens"], "labels": example["labels"]})

    elif "text" in cols and "bio_tags" in cols:
        # Format: raw text + pre-computed BIO tags
        print("  -> Detected text+bio_tags format")
        for example in ds:
            rows.append({"text": example["text"], "bio_tags": example["bio_tags"]})

    elif "tokens" in cols and "ner_tags" in cols:
        # Common HuggingFace NER format (CoNLL-style)
        print("  -> Detected tokens+ner_tags format")
        # Try to recover label names from dataset features
        label_names = None
        if hasattr(ds.features.get("ner_tags", None), "feature") and hasattr(
            ds.features["ner_tags"].feature, "names"
        ):
            label_names = ds.features["ner_tags"].feature.names
        for example in ds:
            tokens = example["tokens"]
            tag_ids = example["ner_tags"]
            if label_names:
                labels = [label_names[t] for t in tag_ids]
            else:
                labels = [id2label.get(t, "O") for t in tag_ids]
            rows.append({"tokens": tokens, "labels": labels})

    else:
        print(f"  ⚠ Unsupported column set: {cols}")
        print("  Falling back to text-based format — expecting 'text' column")
        for example in ds:
            rows.append({"text": example["text"], "bio_tags": ["O"]})

    df = pd.DataFrame(rows)
    print(f"  -> Total samples: {len(df)}")

    # Show label distribution
    if "labels" in df.columns:
        all_labels = [l for sublist in df["labels"] for l in sublist]
        from collections import Counter
        dist = Counter(all_labels)
        for label_name in LABELS:
            count = dist.get(label_name, 0)
            if count > 0:
                print(f"     {label_name:>15s}: {count:>10d}")

    # Split
    train_df, eval_df = train_test_split(
        df, test_size=test_size, random_state=seed
    )

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_df, preserve_index=False)

    print(f"  -> Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    return train_dataset, eval_dataset


# ── CLI ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ModernBERT for PII detection")

    # Model
    p.add_argument("--model", type=str, default="answerdotai/ModernBERT-large",
                    help="HuggingFace repo or local path")
    p.add_argument("--resume", type=str, default=None,
                    help="Resume from checkpoint path")

    # Dataset
    p.add_argument("--dataset", type=str, default="ai4privacy/pii-masking-openpii-1m",
                    help="HuggingFace dataset repo for PII detection")
    p.add_argument("--test-size", type=float, default=0.15,
                    help="Fraction of data for eval split")
    p.add_argument("--seed", type=int, default=3407)

    # Training
    p.add_argument("--output-dir", type=str, default="outputs/modernbert_pii")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=8,
                    help="Gradient accumulation steps (effective batch = batch_size * accum)")
    p.add_argument("--lr", type=float, default=1e-5, help="Peak learning rate")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--logging-steps", type=int, default=40)
    p.add_argument("--eval-steps", type=int, default=480)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-checkpoint", action="store_true", default=True)
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")

    # Class weights
    p.add_argument("--o-weight", type=float, default=0.2,
                    help="Weight for O (outside) class in cross-entropy loss")

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    from mlx_trainer.load import load_token_classification
    from mlx_trainer.args import TrainingArgs
    from mlx_trainer.token_classification_trainer import TokenClassificationTrainer

    print("=" * 70)
    print("Fine-Tuning ModernBERT-large — PII Detection (Token Classification)")
    print("=" * 70)
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs:  {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"  Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"  O-class weight: {args.o_weight}")

    # 1. Load dataset
    train_dataset, eval_dataset = load_pii_dataset(
        args.dataset, seed=args.seed, test_size=args.test_size
    )

    # 2. Load model
    num_labels = len(LABELS)
    print(f"\n[2/4] Loading model: {args.model} ({num_labels} labels)...")

    model_config = {}
    if args.resume:
        model_config["resume_from_checkpoint"] = args.resume

    model, tokenizer = load_token_classification(
        args.model,
        model_config=model_config if model_config else None,
        train=True,
        num_labels=num_labels,
        id2label=id2label,
    )

    # 3. Training args
    print("\n[3/4] Configuring training...")
    training_args = TrainingArgs(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        grad_checkpoint=args.grad_checkpoint,
        resume_from_checkpoint=args.resume,
    )

    # 4. Build class weights — downweight O to handle imbalance
    class_weights = [1.0] * num_labels
    class_weights[label2id["O"]] = args.o_weight
    print(f"  Class weights: O={args.o_weight}, entities=1.0")

    # 5. Train
    print("\n[4/4] Starting training...")
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
    print(f"\n✓ Training complete! Model saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
