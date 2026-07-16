#!/usr/bin/env python3
"""Standalone test for TokenClassificationCollator — avoids heavy package imports."""

import importlib.util
import os
import sys

# Direct-load the collator module, bypassing __init__.py
_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "tcc",
    os.path.join(_dir, "..", "mlx_trainer", "token_classification_collator.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
TokenClassificationCollator = mod.TokenClassificationCollator


# ── Fake tokenizer / encoding ──────────────────────────────────────────────

class FakeEncoding:
    def __init__(self, input_ids, word_ids):
        self._input_ids = input_ids
        self._word_ids = word_ids

    def __getitem__(self, key):
        if key == "input_ids":
            return self._input_ids
        if key == "attention_mask":
            return [1] * len(self._input_ids)
        raise KeyError(key)

    def word_ids(self):
        return self._word_ids


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0

    def __call__(self, words, padding=False, truncation=True, max_length=512, is_split_into_words=True):
        input_ids = [101]  # [CLS]
        word_ids_list = [None]

        for i, word in enumerate(words):
            token_id = 10 + i
            input_ids.append(token_id)
            word_ids_list.append(i)

            # Simulate subword split for "Alvaro" → ["Al", "##varo"]
            if word == "Alvaro":
                input_ids.append(200)
                word_ids_list.append(i)

        input_ids.append(102)  # [SEP]
        word_ids_list.append(None)
        return FakeEncoding(input_ids, word_ids_list)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_basic_alignment():
    label2id = {"O": 0, "B-PER": 1, "I-PER": 2}
    collator = TokenClassificationCollator(FakeTokenizer(), max_length=32, label2id=label2id)

    features = [
        {"tokens": ["Alvaro", "is", "here"], "labels": ["B-PER", "O", "O"]}
    ]
    batch = collator(features)
    labels = batch["labels"][0].tolist()

    # [CLS]→-100, Al→B-PER(1), ##varo→I-PER(2), is→O(0), here→O(0), [SEP]→-100
    assert labels[0] == -100, f"CLS: {labels[0]}"
    assert labels[1] == 1, f"Al: {labels[1]}"
    assert labels[2] == 2, f"##varo: {labels[2]}"
    assert labels[3] == 0, f"is: {labels[3]}"
    assert labels[4] == 0, f"here: {labels[4]}"
    assert labels[5] == -100, f"SEP: {labels[5]}"
    print("PASS: test_basic_alignment")


def test_text_format():
    label2id = {"O": 0, "B-PER": 1, "I-PER": 2}
    collator = TokenClassificationCollator(FakeTokenizer(), max_length=32, label2id=label2id)

    features = [
        {"text": "Alvaro is here", "bio_tags": ["B-PER", "O", "O"]}
    ]
    batch = collator(features)
    labels = batch["labels"][0].tolist()

    assert labels[0] == -100
    assert labels[1] == 1
    assert labels[2] == 2
    assert labels[3] == 0
    assert labels[4] == 0
    assert labels[5] == -100
    print("PASS: test_text_format")


def test_batch_padding():
    label2id = {"O": 0, "B-PER": 1, "I-PER": 2}
    collator = TokenClassificationCollator(FakeTokenizer(), max_length=32, label2id=label2id)

    features = [
        {"tokens": ["hi"], "labels": ["O"]},
        {"tokens": ["Alvaro", "is", "here"], "labels": ["B-PER", "O", "O"]},
    ]
    batch = collator(features)

    len1 = len(batch["input_ids"][0].tolist())
    len2 = len(batch["input_ids"][1].tolist())
    assert len1 == len2, f"Lengths differ: {len1} != {len2}"

    # Short item: CLS(3 tokens: CLS,hi,SEP=3). Padding labels should be -100
    labels1 = batch["labels"][0].tolist()
    pad_labels = labels1[3:]
    assert all(l == -100 for l in pad_labels), f"Pad labels: {pad_labels}"
    print("PASS: test_batch_padding")


def test_o_continuation():
    """O label → stays O (not I-O) on continuation subwords."""
    label2id = {"O": 0, "B-PER": 1, "I-PER": 2}

    class SplitTokenizer:
        pad_token_id = 0
        def __call__(self, words, **kw):
            ids, wids = [101], [None]
            for i, w in enumerate(words):
                ids.append(10 + i); wids.append(i)
                if w == "email":
                    ids.append(200); wids.append(i)
            ids.append(102); wids.append(None)
            return FakeEncoding(ids, wids)

    collator = TokenClassificationCollator(SplitTokenizer(), max_length=32, label2id=label2id)
    features = [{"tokens": ["my", "email", "test"], "labels": ["O", "O", "O"]}]
    batch = collator(features)
    labels = batch["labels"][0].tolist()

    # CLS→-100, my→O, email→O, ##il→O, test→O, SEP→-100
    assert labels[3] == 0, f"Continuation of O should be 0, got {labels[3]}"
    print("PASS: test_o_continuation")


def test_ixxx_continuation():
    """I-PER subword continuation stays I-PER."""
    label2id = {"O": 0, "B-PER": 1, "I-PER": 2}

    class SplitTokenizer:
        pad_token_id = 0
        def __call__(self, words, **kw):
            ids, wids = [101], [None]
            for i, w in enumerate(words):
                ids.append(10 + i); wids.append(i)
                if w == "González":
                    ids.append(200); wids.append(i)
            ids.append(102); wids.append(None)
            return FakeEncoding(ids, wids)

    collator = TokenClassificationCollator(SplitTokenizer(), max_length=32, label2id=label2id)
    features = [{"tokens": ["Señor", "González"], "labels": ["B-PER", "I-PER"]}]
    batch = collator(features)
    labels = batch["labels"][0].tolist()

    # CLS→-100, Señor→B-PER(1), Gon→I-PER(2), ##zález→I-PER(2), SEP→-100
    assert labels[2] == 2, f"I-PER continuation: {labels[2]}"
    print("PASS: test_ixxx_continuation")


if __name__ == "__main__":
    test_basic_alignment()
    test_text_format()
    test_batch_padding()
    test_o_continuation()
    test_ixxx_continuation()
    print("\nAll tests passed!")
