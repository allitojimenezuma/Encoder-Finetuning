"""
Quick smoke test for PII inference — uses the model trained by test_train.py.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

from mlx_trainer.load import load_token_classification
from examples.pii.inference_pii import run_inference, print_results

# ── Must match the 9-label schema used in test_train.py ───────────
LABELS = [
    "O", "B-PER", "I-PER", "B-EMAIL", "I-EMAIL",
    "B-PHONE", "I-PHONE", "B-ADDRESS", "I-ADDRESS",
]
id2label = {i: l for i, l in enumerate(LABELS)}

# ── Tiny hardcoded texts ──────────────────────────────────────────
TESTS = [
    "My name is Alice and I live in New York",
    "Contact Bob at bob@test.com or call 555-1234",
    "Hello world, no PII here",
]

print("=" * 60)
print("PII Inference Smoke Test")
print("=" * 60)

# ── Load model ────────────────────────────────────────────────────
print("\n[1/2] Loading model from outputs/modernbert_pii_test ...")
model, tokenizer = load_token_classification(
    "outputs/modernbert_pii_test/checkpoint-40",
    num_labels=len(LABELS),
    id2label=id2label,
)

# ── Run inference ─────────────────────────────────────────────────
print("\n[2/2] Running inference...")
for text in TESTS:
    entities, highlighted = run_inference(model, tokenizer, text)
    print_results(text, entities, highlighted)

print("\n✓ Smoke test passed — inference runs end-to-end.")
