"""
ModernBERT PII Detection — Inference Script

Detects PII entities (names, emails, phones, etc.) in text using
a fine-tuned token classification model with BIO labeling.

Usage:
    python examples/pii/inference_pii.py --text "My name is Alice and my email is alice@example.com"
    python examples/pii/inference_pii.py --file input.txt
    python examples/pii/inference_pii.py --json-input data.json --model outputs/modernbert_pii/checkpoint-3941
"""
import argparse
import json
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from mlx_trainer.load import load_token_classification


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


# ── ANSI colors for terminal output ────────────────────────────────
ENTITY_COLORS = {
    "PER":         "\033[93m",  # yellow
    "EMAIL":       "\033[94m",  # blue
    "PHONE":       "\033[92m",  # green
    "IBAN":        "\033[95m",  # magenta
    "CREDIT_CARD": "\033[91m",  # red
    "IP_ADDRESS":  "\033[96m",  # cyan
    "ADDRESS":     "\033[97m",  # white (bright)
    "SSN":         "\033[31m",  # dark red
    "USERNAME":    "\033[33m",  # orange
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def get_entity_type(bio_label: str) -> Optional[str]:
    """Extract entity type from BIO label (e.g. 'B-PER' -> 'PER')."""
    if bio_label == "O" or not bio_label.startswith(("B-", "I-")):
        return None
    return bio_label[2:]


def extract_entities(
    tokens: List[str],
    predictions: List[int],
    confidence: List[float],
    threshold: float = 0.5,
) -> List[Dict]:
    """Convert per-token BIO predictions to entity spans.

    Returns list of dicts with keys:
        text, label, start_token, end_token, confidence
    """
    entities = []
    current_entity_type = None
    current_tokens = []
    current_confs = []

    for i, (token, pred_id, conf) in enumerate(zip(tokens, predictions, confidence)):
        label = id2label.get(pred_id, "O")
        entity_type = get_entity_type(label)

        if label.startswith("B-"):
            # Close previous entity if any
            if current_entity_type and current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "label": current_entity_type,
                    "start_token": i - len(current_tokens),
                    "end_token": i,
                    "confidence": sum(current_confs) / len(current_confs),
                })
            current_entity_type = entity_type
            current_tokens = [token]
            current_confs = [conf]
        elif label.startswith("I-") and entity_type == current_entity_type:
            # Continuation of current entity
            current_tokens.append(token)
            current_confs.append(conf)
        else:
            # O label or entity type mismatch — close current entity
            if current_entity_type and current_tokens:
                entities.append({
                    "text": " ".join(current_tokens),
                    "label": current_entity_type,
                    "start_token": i - len(current_tokens),
                    "end_token": i,
                    "confidence": sum(current_confs) / len(current_confs),
                })
                current_entity_type = None
                current_tokens = []
                current_confs = []

    # Close trailing entity
    if current_entity_type and current_tokens:
        entities.append({
            "text": " ".join(current_tokens),
            "label": current_entity_type,
            "start_token": len(tokens) - len(current_tokens),
            "end_token": len(tokens),
            "confidence": sum(current_confs) / len(current_confs),
        })

    return entities


def format_highlighted_text(
    text: str,
    entities: List[Dict],
    use_color: bool = True,
) -> str:
    """Reconstruct original text with highlighted entity spans."""
    if not entities:
        return text

    sorted_entities = sorted(entities, key=lambda e: e["start_token"])

    highlighted_parts = []
    last_end = 0

    # Find character-level positions
    char_entities = _find_char_positions(text, sorted_entities)

    for start, end, label, conf in char_entities:
        if start < last_end:
            continue

        highlighted_parts.append(text[last_end:start])
        entity_text = text[start:end]

        if use_color:
            color = ENTITY_COLORS.get(label, "\033[97m")
            highlighted_parts.append(
                f"{color}{BOLD}[{label}]{RESET} {color}{entity_text}{RESET}"
            )
        else:
            highlighted_parts.append(f"[{label}] {entity_text}")

        last_end = end

    highlighted_parts.append(text[last_end:])
    return "".join(highlighted_parts)


def _find_char_positions(
    text: str,
    entities: List[Dict],
) -> List[Tuple[int, int, str, float]]:
    """Find character-level positions of entities in original text."""
    result = []
    for ent in entities:
        entity_text = ent["text"]
        idx = text.find(entity_text)
        if idx >= 0:
            result.append((idx, idx + len(entity_text), ent["label"], ent["confidence"]))
        else:
            clean = entity_text.replace("▁", "").strip()
            if clean:
                idx = text.find(clean)
                if idx >= 0:
                    result.append((idx, idx + len(clean), ent["label"], ent["confidence"]))
    return result


def run_inference(
    model,
    tokenizer,
    text: str,
    threshold: float = 0.5,
    max_length: int = 512,
) -> Tuple[List[Dict], str]:
    """Run PII inference on a single text string.

    Returns (entities, highlighted_text).
    """
    inputs = tokenizer._tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length,
    )

    input_ids = mx.array(inputs["input_ids"])
    attention_mask = mx.array(inputs["attention_mask"])

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    probabilities = outputs["probabilities"]
    pred_ids = mx.argmax(probabilities, axis=-1).tolist()[0]
    probs = probabilities.tolist()[0]

    confidences = [probs[i][pred_ids[i]] for i in range(len(pred_ids))]

    try:
        tokens = tokenizer._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    except AttributeError:
        tokens = [str(t) for t in inputs["input_ids"][0]]

    entities = extract_entities(tokens, pred_ids, confidences, threshold)
    highlighted = format_highlighted_text(text, entities)

    return entities, highlighted


def print_results(text: str, entities: List[Dict], highlighted: str):
    """Print formatted inference results."""
    print("\n" + "=" * 70)
    print(f"{BOLD}PII DETECTION RESULTS{RESET}")
    print("=" * 70)

    if not entities:
        print(f"\n  {DIM}No PII detected.{RESET}\n")
        print(f"  Text: {text}")
        print("=" * 70)
        return

    print(f"\n  {BOLD}Text with PII:{RESET}")
    print(f"  {highlighted}")

    print(f"\n  {BOLD}Detected Entities:{RESET}")
    print(f"  {'Label':<16} {'Confidence':>10}  {'Text'}")
    print(f"  {'-'*16} {'-'*10}  {'-'*30}")
    for ent in entities:
        conf_pct = ent["confidence"] * 100
        print(f"  {ent['label']:<16} {conf_pct:>9.1f}%  {ent['text']}")

    print(f"\n  {BOLD}Summary:{RESET} {len(entities)} PII entity(ies) found")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(
        description="PII Detection Inference — ModernBERT Token Classification"
    )

    p.add_argument(
        "--model", type=str,
        default="outputs/modernbert_pii",
        help="Path to trained model checkpoint or HuggingFace repo",
    )

    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Input text to analyze")
    input_group.add_argument("--file", type=str, help="Path to text file to analyze")
    input_group.add_argument("--json-input", type=str,
                             help="JSON file with 'texts' key (list of strings)")

    p.add_argument("--threshold", type=float, default=0.5,
                    help="Minimum confidence threshold for entity extraction")
    p.add_argument("--max-length", type=int, default=512,
                    help="Max token length")
    p.add_argument("--no-color", action="store_true",
                    help="Disable colored terminal output")

    return p.parse_args()


def main():
    args = parse_args()
    num_labels = len(LABELS)

    print("=" * 70)
    print(f"{BOLD}ModernBERT PII Detection — Inference{RESET}")
    print("=" * 70)
    print(f"  Model:     {args.model}")
    print(f"  Labels:    {num_labels}")
    print(f"  Threshold: {args.threshold}")

    # 1. Load model
    print(f"\n[1/3] Loading model from '{args.model}'...")
    model, tokenizer = load_token_classification(
        args.model,
        num_labels=num_labels,
        id2label=id2label,
    )

    # 2. Prepare inputs
    texts = []
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, "r") as f:
            texts = [f.read().strip()]
        print(f"  Loaded text from: {args.file}")
    elif args.json_input:
        with open(args.json_input, "r") as f:
            data = json.load(f)
        texts = data["texts"]
        print(f"  Loaded {len(texts)} texts from: {args.json_input}")

    # 3. Run inference
    print(f"\n[2/3] Running inference on {len(texts)} input(s)...")
    for i, text in enumerate(texts):
        if len(texts) > 1:
            print(f"\n--- Input {i+1}/{len(texts)} ---")
        entities, highlighted = run_inference(
            model, tokenizer, text,
            threshold=args.threshold,
            max_length=args.max_length,
        )
        print_results(text, entities, highlighted)

    print("\n[3/3] Done.")


if __name__ == "__main__":
    main()
