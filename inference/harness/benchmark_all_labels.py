"""
benchmark_all_labels.py

The authoritative benchmark for the /goal task: per-label P/R/F1 across
all 7 entity types, on the two clean held-out test sets (8-file hard +
56-file majority, 64 files total, zero training exposure for any model
variant). Uses span_matcher's match_entities_fixed (the corrected
matching logic) and WITH postprocessing (the production configuration).

Usage:
  python3 benchmark_all_labels.py <adapter_dir_or_None_for_baseline>
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

sys.path.insert(0, ".")
from evaluate_finetuned import (  # noqa: E402
    DATA_DIR, MODEL_PATH, FINETUNED_LABELS, SYNTHETIC_LABELS, run_fulltext, load_test_cases,
)
from evaluate_majority import load_majority_test_cases  # noqa: E402
from span_matcher import match_entities_fixed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "presidio-gliner" / "scoring_utils"))
from eval_pipeline import ORIG_DIR, GOLD_DIR, extract_gold_entities, _prf  # noqa: E402

from gliner2 import GLiNER2

THRESHOLD = 0.35
CANON_MERGE = {"SG_ADDRESS_BLOCK_NUMBER": "SG_ADDRESS_BLOCK", "SG_ADDRESS_UNIT_NUMBER": "SG_ADDRESS_UNIT"}

# The 7 base labels reported by this benchmark.
BASE_REPORT_LABELS = ["EMAIL_ADDRESS", "SG_ADDRESS", "SG_ADDRESS_BLOCK", "SG_ADDRESS_UNIT",
                      "SG_NRIC_FIN", "SG_PHONE_NUMBER", "SG_POSTAL_CODE"]
# The 2 labels added by the synthetic corpus. Reported only in --synthetic mode;
# see the note in main() about gold availability.
SYNTHETIC_EXTRA_REPORT_LABELS = ["ACCOUNT_NUMBER", "FULL_NAME"]

# Only full-format NRIC/FIN is scored as gold; partial 3/4-digit+letter fragments
# are ignored (a full NRIC is [STFG] + 7 digits + a checksum letter).
import re  # noqa: E402
_FULL_NRIC_RE = re.compile(r'^[STFG]\d{7}[A-Z]$')


def _is_full_nric_gold(text):
    stripped = re.sub(r'[\s\-]', '', text).upper()
    return bool(_FULL_NRIC_RE.match(stripped))


def _filter_gold(gold):
    return [(t, l) for t, l in gold if l != "SG_NRIC_FIN" or _is_full_nric_gold(t)]


def build_benchmark_cases():
    test_files = Path(DATA_DIR / "test_files.txt").read_text().strip().splitlines()
    hard8_cases = load_test_cases(test_files)
    hard8_text = [(" ".join(rows), _filter_gold(gold)) for _f, rows, gold in hard8_cases]
    majority_cases = [(text, _filter_gold(gold)) for text, gold in load_majority_test_cases()]
    return hard8_text + majority_cases


def build_synthetic_cases():
    """Held-out test split of the synthetic corpus (test_mixed2.jsonl, built
    by build_synthetic_training_data.py). Gold there is a flat list of strings
    per lower-case label, so it is canonicalised here to the same
    [(text, CANON_LABEL), ...] shape the other benchmark sets use.

    The SG_NRIC_FIN full-format gold filter is deliberately NOT applied here.

    Rationale (decided 2026-07-21): the synthetic corpus intentionally tags NRIC
    fragments split across a pause (e.g. 'S-1-2-3-4' + '5-6-7-A' alongside the
    joined 'S1234567A'), and the model is TRAINED on all of them. Applying
    _filter_gold() would discard 359 of 586 test-set NRIC gold entries (61%),
    so every fragment the model correctly recovers would be scored as a false
    positive -- penalising it for exactly the behaviour the corpus teaches.

    The legacy path (build_benchmark_cases) still applies _filter_gold(), so the
    real-data benchmark numbers remain directly comparable to the existing
    309-example checkpoint. The two eval paths are each internally consistent."""
    from evaluate_finetuned import CANON  # noqa: PLC0415  (local: keeps module import surface unchanged)
    path = Path(__file__).resolve().parents[2] / "data" / "frozen" / "test_gold_419.jsonl"
    cases = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            gold = [(t, CANON[label])
                    for label, values in d["output"]["entities"].items()
                    for t in values]
            cases.append((d["input"], gold))
    return cases


def evaluate(model, cases, labels=FINETUNED_LABELS, windowed=False):
    from evaluate_finetuned import run_windowed  # noqa: PLC0415
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
    args = [a for a in sys.argv[1:]]
    synthetic = "--synthetic" in args
    if synthetic:
        args.remove("--synthetic")
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

    if synthetic:
        # 9-label mode, scored on the held-out synthetic test split. The legacy
        # hard8+majority gold has NO account_number/full_name annotations, so
        # those two labels can only be measured here.
        cases = build_synthetic_cases()
        query_labels = SYNTHETIC_LABELS
        all_labels = BASE_REPORT_LABELS + SYNTHETIC_EXTRA_REPORT_LABELS
        print(f"Benchmark set: {len(cases)} files (held-out synthetic test split, 9 labels)")
    else:
        cases = build_benchmark_cases()
        query_labels = FINETUNED_LABELS
        all_labels = BASE_REPORT_LABELS
        print(f"Benchmark set: {len(cases)} files (8-file hard + 56-file majority)")

    if windowed:
        print("Inference: OVERLAPPING WINDOWS (1800-char windows, 400 overlap)")
    per_label_tp, per_label_fp, per_label_fn, total_tp, total_fp, total_fn = evaluate(
        model, cases, query_labels, windowed=windowed
    )

    print()
    print("=" * 90)
    print("PER-LABEL BENCHMARK RESULTS")
    print("=" * 90)
    all_pass = True
    for label in all_labels:
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
    print(f"ALL {len(all_labels)} LABELS PASS (P>0.8, R>0.8, F1>0.8): {all_pass}")


if __name__ == "__main__":
    main()
