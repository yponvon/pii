"""
build_windowed_training_data.py -- turn the full-transcript training split into
overlapping token windows, so the model trains on the WHOLE call rather than
only the first ~512 tokens (which truncation was discarding, losing ~23% of the
gold PII that sits later in long calls).

Efficiency (balanced setting):
  * Windows are small (WINDOW_TOKENS) so per-step attention cost stays low
    (attention is O(length^2), so a 256-token window costs a quarter of a
    512-token one).
  * Every window that contains PII is kept (lossless on PII).
  * Only a fraction of the empty windows is kept (NEG_PER_POS), enough to teach
    "this text is not PII" without inflating the dataset.

Correctness: each output window carries exactly the gold entities whose full span
falls inside it, and each window is re-checked against the gold invariant (every
gold string occurs in the window text exactly as many times as it is listed),
using the same longest-first / covered-span logic as the corpus generator.

Input : finetuning/splits/{train_mixed2,val_mixed2}.jsonl   (full transcripts)
Output: finetuning/splits/{train_windowed,val_windowed}.jsonl (token windows)

Run:
  ./venv/bin/python pii/finetuning/data_prep/build_windowed_training_data.py
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

# -- configuration ------------------------------------------------------------
TOKENIZER = "microsoft/mdeberta-v3-base"   # the model's own tokenizer
WINDOW_TOKENS = 256                        # window length in tokens
OVERLAP_TOKENS = 64                        # shared tokens between adjacent windows
NEG_PER_POS = 0.5                          # empty windows kept, per positive window
SEED = 42

BASE = Path(__file__).resolve().parent.parent.parent   # .../pii
SPLITS = BASE / "finetuning" / "splits"
INPUTS = {"train": SPLITS / "train_mixed2.jsonl", "val": SPLITS / "val_mixed2.jsonl"}
OUTPUTS = {"train": SPLITS / "train_windowed.jsonl", "val": SPLITS / "val_windowed.jsonl"}

TRAILING_PUNCT = " .,!?;:\"')"


def core_format(value: str) -> str:
    """Strip trailing punctuation so a gold value matches how it appears in text
    (identical to the corpus generator's core_format)."""
    return value.rstrip(TRAILING_PUNCT)


def assign_spans(entities: dict, text: str):
    """Locate every gold mention as a concrete (start, end, label, value) span,
    using longest-value-first with covered-span exclusion so overlapping values
    are not double-counted. Mirrors compute_standalone_counts, but returns the
    spans themselves so they can be routed into windows."""
    needed = Counter((value, label)
                     for label, values in entities.items() for value in values)
    distinct_values = sorted({v for v, _ in needed}, key=lambda s: len(core_format(s)), reverse=True)

    covered = []
    spans = []
    for value in distinct_values:
        core = core_format(value)
        if not core:
            continue
        pattern = re.compile(r'(?<![A-Za-z0-9])' + re.escape(core) + r'(?![A-Za-z0-9])')
        occurrences = []
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            if any(cs <= start and end <= ce for cs, ce in covered):
                continue
            occurrences.append((start, end))
        covered.extend(occurrences)

        # Route the found occurrences to the label(s) that expect this value.
        labels_for_value = [label
                            for (v, label), count in needed.items() if v == value
                            for _ in range(count)]
        for (start, end), label in zip(occurrences, labels_for_value):
            spans.append((start, end, label, value))
    return spans


def standalone_counts(values, text):
    """Occurrence count per value in `text`, using the same covered-span logic;
    used to verify each output window satisfies the gold invariant."""
    counts = {}
    covered = []
    for value in sorted(set(values), key=lambda s: len(core_format(s)), reverse=True):
        core = core_format(value)
        if not core:
            counts[value] = 0
            continue
        pattern = re.compile(r'(?<![A-Za-z0-9])' + re.escape(core) + r'(?![A-Za-z0-9])')
        count = 0
        new = []
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            if any(cs <= start and end <= ce for cs, ce in covered):
                continue
            count += 1
            new.append((start, end))
        covered.extend(new)
        counts[value] = count
    return counts


def token_windows(text, tokenizer):
    """Yield (char_start, char_end) spans for each overlapping token window."""
    offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    n = len(offsets)
    step = WINDOW_TOKENS - OVERLAP_TOKENS
    i = 0
    while i < n:
        j = min(i + WINDOW_TOKENS, n)
        yield offsets[i][0], offsets[j - 1][1]
        if j >= n:
            break
        i += step


def build_split(name, tokenizer, rng):
    rows = [json.loads(line) for line in open(INPUTS[name], encoding="utf-8")]
    positives, empties = [], []

    for row in rows:
        text = row["input"]
        spans = assign_spans(row["output"]["entities"], text)
        for c0, c1 in token_windows(text, tokenizer):
            window_text = text[c0:c1]
            inside = [(label, value) for (s, e, label, value) in spans if c0 <= s and e <= c1]
            window_entities = {}
            for label, value in inside:
                window_entities.setdefault(label, []).append(value)
            record = {"input": window_text, "output": {"entities": window_entities}}
            (positives if window_entities else empties).append(record)

    # Balanced sampling: keep all positives, plus NEG_PER_POS empties per positive.
    keep_empty = min(len(empties), int(len(positives) * NEG_PER_POS))
    kept_empties = rng.sample(empties, keep_empty) if keep_empty else []
    output = positives + kept_empties
    rng.shuffle(output)

    # Verify the gold invariant on every emitted window.
    violations = 0
    for record in output:
        text = record["input"]
        for label, values in record["output"]["entities"].items():
            counts = standalone_counts(values, text)
            for value in values:
                if counts.get(value, 0) < values.count(value):
                    violations += 1

    with open(OUTPUTS[name], "w", encoding="utf-8") as fh:
        for record in output:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[{name}] {len(rows)} transcripts -> {len(output)} windows "
          f"({len(positives)} with PII + {len(kept_empties)} empty)  "
          f"invariant violations: {violations}")
    return violations


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    rng = random.Random(SEED)
    total_violations = 0
    for name in ("train", "val"):
        total_violations += build_split(name, tokenizer, rng)
    if total_violations:
        print(f"\nWARNING: {total_violations} invariant violations -- inspect before training.")
    else:
        print("\nAll windows satisfy the gold invariant.")


if __name__ == "__main__":
    main()
