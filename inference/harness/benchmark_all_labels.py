"""
benchmark_all_labels.py

Per-label P/R/F1 for the fine-tuned model across all 9 entity types, scored on
the frozen 419-file held-out test set (data/frozen/test_gold_419.jsonl, zero
training exposure). Uses span_matcher.match_entities_fixed (the benchmark
matcher) with the production postprocessing pipeline.

This is the per-label companion to run_frozen_comparison.py, which produces the
three-way (baseline / rule-based / fine-tuned) overall comparison on the same
frozen set.

Usage:
  python3 benchmark_all_labels.py [adapter_dir | None-for-baseline] [--windowed]
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

from gliner2 import GLiNER2

HARNESS_DIR = Path(__file__).resolve().parent
PII_ROOT = HARNESS_DIR.parents[1]
for _p in (HARNESS_DIR, PII_ROOT / "utils"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluate_finetuned import (  # noqa: E402
    MODEL_PATH, SYNTHETIC_LABELS, CANON, run_fulltext, run_windowed,
)
from span_matcher import match_entities_fixed  # noqa: E402
from scoring import _prf  # noqa: E402

THRESHOLD = 0.35
CANON_MERGE = {"SG_ADDRESS_BLOCK_NUMBER": "SG_ADDRESS_BLOCK", "SG_ADDRESS_UNIT_NUMBER": "SG_ADDRESS_UNIT"}

# The 9 reported labels: the 7 base labels plus the 2 this project added.
BASE_REPORT_LABELS = ["EMAIL_ADDRESS", "SG_ADDRESS", "SG_ADDRESS_BLOCK", "SG_ADDRESS_UNIT",
                      "SG_NRIC_FIN", "SG_PHONE_NUMBER", "SG_POSTAL_CODE"]
NEW_REPORT_LABELS = ["ACCOUNT_NUMBER", "FULL_NAME"]
ALL_REPORT_LABELS = BASE_REPORT_LABELS + NEW_REPORT_LABELS


def build_cases():
    """Load the frozen held-out test set (419 authentic transcripts).

    Gold is a flat list of surface strings per lower-case label; this
    canonicalises it to the [(text, CANON_LABEL), ...] shape the matcher expects.

    NRIC gold is kept as-is (fragments included): the corpus intentionally tags
    NRIC pieces split across a pause, and the model is trained on them, so
    filtering to full-format NRIC only would score correct fragment catches as
    false positives.
    """
    path = PII_ROOT / "data" / "frozen" / "test_gold_419.jsonl"
    cases = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            gold = [(t, CANON[label])
                    for label, values in d["output"]["entities"].items()
                    for t in values]
            cases.append((d["input"], gold))
    return cases


def evaluate(model, cases, labels, windowed=False):
    per_label_tp = defaultdict(int)
    per_label_fp = defaultdict(int)
    per_label_fn = defaultdict(int)
    total_tp = total_fp = total_fn = 0

    for full_text, gold in cases:
        if windowed:
            pred = run_windowed(model, full_text, labels, THRESHOLD)
        else:
            pred = run_fulltext(model, full_text, labels, THRESHOLD)
        tp, fp, fn, errors_fp, errors_fn, tp_by_label = match_entities_fixed(pred, gold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        for label, count in tp_by_label.items():
            per_label_tp[CANON_MERGE.get(label, label)] += count
        for label, _text in errors_fp:
            per_label_fp[CANON_MERGE.get(label, label)] += 1
        for label, _text in errors_fn:
            per_label_fn[CANON_MERGE.get(label, label)] += 1

    return per_label_tp, per_label_fp, per_label_fn, total_tp, total_fp, total_fn


def main():
    args = list(sys.argv[1:])
    windowed = "--windowed" in args
    if windowed:
        args.remove("--windowed")
    adapter_dir = args[0] if args else None

    model = GLiNER2.from_pretrained(MODEL_PATH)
    if adapter_dir and adapter_dir != "None":
        model.load_adapter(adapter_dir)
        print(f"Loaded adapter: {adapter_dir}")
    else:
        print("Using BASE model (no adapter -- zero-shot baseline)")

    cases = build_cases()
    print(f"Benchmark set: {len(cases)} files (frozen held-out test, 9 labels)")
    if windowed:
        print("Inference: OVERLAPPING WINDOWS (1800-char windows, 400 overlap)")

    per_label_tp, per_label_fp, per_label_fn, total_tp, total_fp, total_fn = evaluate(
        model, cases, SYNTHETIC_LABELS, windowed=windowed
    )

    print()
    print("=" * 90)
    print("PER-LABEL BENCHMARK RESULTS")
    print("=" * 90)
    all_pass = True
    for label in ALL_REPORT_LABELS:
        tp = per_label_tp[label]
        fp = per_label_fp[label]
        fn = per_label_fn[label]
        p, r, f = _prf(tp, fp, fn)
        gate = "PASS" if (p > 0.8 and r > 0.8 and f > 0.8) else "FAIL"
        if gate == "FAIL":
            all_pass = False
        print(f"{label:<20} TP={tp:>4} FP={fp:>4} FN={fn:>4}  P={p:.4f}  R={r:.4f}  F1={f:.4f}  [{gate}]")

    p, r, f = _prf(total_tp, total_fp, total_fn)
    print()
    print(f"OVERALL: TP={total_tp} FP={total_fp} FN={total_fn}  P={p:.4f}  R={r:.4f}  F1={f:.4f}")
    print()
    print(f"ALL {len(ALL_REPORT_LABELS)} LABELS PASS (P>0.8, R>0.8, F1>0.8): {all_pass}")


if __name__ == "__main__":
    main()
