"""
eval_pipeline_twopass.py

Experiment: two-pass phone vs. account/reference-number disambiguation on top
of the existing full-text pipeline (eval_pipeline.py).

Pass 1 (unchanged): run_model() from eval_pipeline.py -- full-text detection
with NRIC/postal/content-filter postprocessing already in place.

Pass 2 (new): for every span Pass 1 labelled SG_PHONE_NUMBER, take a narrow
+/-150 char window around just that span and re-query GLiNER2 with a
restricted, confusable label set:
    positive: sg_phone_number, sg_contact_number
    negative: account_number, reference_number, case_number, invoice_number
If a negative label scores higher than any positive label for the same
local span, the candidate is suppressed (treated as an account/reference
number, not a phone number). Otherwise it is kept.

This never introduces new spans -- it can only remove Pass-1 phone
candidates that a narrow, label-restricted re-check disagrees with.

Usage:
  /Users/yvonne/INTERN/echolens/venv/bin/python3 eval_pipeline_twopass.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from eval_pipeline import (
    ORIG_DIR,
    GOLD_DIR,
    THRESHOLD,
    LABELS,
    model,
    extract_gold_entities,
    match_entities,
    _prf,
    _is_valid_nric,
    _passes_content_filter,
    _find_postal_codes_with_context,
)

OUT_DIR = Path("/Users/yvonne/INTERN/echolens/data_25_june/run3_twopass_eval")

# -- Pass 2 config --------------------------------------------------------

WINDOW_CHARS = 150
RESCORE_THRESHOLD = 0.15  # low threshold -- we just want comparative scores
POSITIVE_LABELS = {"sg_phone_number", "sg_contact_number"}
NEGATIVE_LABELS = {"account_number", "reference_number", "case_number", "invoice_number"}
RESCORE_LABELS = list(POSITIVE_LABELS | NEGATIVE_LABELS)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def _disambiguate_phone_span(full_text: str, start: int, end: int) -> bool:
    """Return True if this SG_PHONE_NUMBER span should be KEPT after local re-check."""
    win_start = max(0, start - WINDOW_CHARS)
    win_end = min(len(full_text), end + WINDOW_CHARS)
    window = full_text[win_start:win_end]
    local_start, local_end = start - win_start, end - win_start

    result = model.extract_entities(
        window,
        RESCORE_LABELS,
        threshold=RESCORE_THRESHOLD,
        include_spans=True,
        include_confidence=True,
    )

    best_pos = 0.0
    best_neg = 0.0
    for label, ents in result.get("entities", {}).items():
        for e in ents:
            if not _overlaps(e["start"], e["end"], local_start, local_end):
                continue
            score = e.get("confidence", 0.0)
            if label in POSITIVE_LABELS:
                best_pos = max(best_pos, score)
            elif label in NEGATIVE_LABELS:
                best_neg = max(best_neg, score)

    return not (best_neg > best_pos)


# -- Pass 1 + spans (mirrors eval_pipeline.run_model, but also returns spans) --

def run_model_pass1(full_text: str):
    result = model.extract_entities(full_text, LABELS, threshold=THRESHOLD, include_spans=True)

    entities: list[tuple[str, str]] = []
    spans: list[tuple[int, int, str]] = []

    for label, ents in result["entities"].items():
        label_key = label.upper()
        for e in ents:
            text = e["text"].strip()
            if not text:
                continue
            if label_key == "SG_NRIC_FIN" and not _is_valid_nric(text):
                continue
            if not _passes_content_filter(text, label_key):
                continue
            entities.append((text, label_key))
            spans.append((e["start"], e["end"], label_key))

    postal_new = _find_postal_codes_with_context(full_text, spans)
    for ms, me, lbl in postal_new:
        entities.append((full_text[ms:me], lbl))
        spans.append((ms, me, lbl))

    return entities, spans


def run_model_twopass(full_text: str):
    entities, spans = run_model_pass1(full_text)

    kept_entities = []
    n_checked = 0
    n_suppressed = 0
    for (text, label), (start, end, _label2) in zip(entities, spans):
        if label == "SG_PHONE_NUMBER":
            n_checked += 1
            if not _disambiguate_phone_span(full_text, start, end):
                n_suppressed += 1
                continue
        kept_entities.append((text, label))

    return kept_entities, n_checked, n_suppressed


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(ORIG_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} files\n")
    print(f"{'File':<55} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6} {'chk':>4} {'sup':>4}")
    print("-" * 100)

    per_file_rows = []
    total_tp = total_fp = total_fn = 0
    total_checked = total_suppressed = 0

    for orig_path in csv_files:
        fname = orig_path.name
        gold_path = GOLD_DIR / fname

        orig_df = pd.read_csv(orig_path)
        gold_df = pd.read_csv(gold_path)

        full_text = " ".join(orig_df["text"].fillna("").astype(str).tolist())
        pred, n_checked, n_suppressed = run_model_twopass(full_text)
        gold = extract_gold_entities(orig_df, gold_df)

        tp, fp, fn, _, _, _ = match_entities(pred, gold)
        p, r, f = _prf(tp, fp, fn)

        print(f"{fname:<55} {tp:>4} {fp:>4} {fn:>4} {p:>6.4f} {r:>6.4f} {f:>6.4f} {n_checked:>4} {n_suppressed:>4}")

        per_file_rows.append([fname, tp, fp, fn, p, r, f, n_checked, n_suppressed])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_checked += n_checked
        total_suppressed += n_suppressed

    p_all, r_all, f_all = _prf(total_tp, total_fp, total_fn)
    print("-" * 100)
    print(f"{'OVERALL':<55} {total_tp:>4} {total_fp:>4} {total_fn:>4} {p_all:>6.4f} {r_all:>6.4f} {f_all:>6.4f} {total_checked:>4} {total_suppressed:>4}")
    print(f"\nPrecision={p_all}  Recall={r_all}  F1={f_all}")
    print(f"Phone candidates checked={total_checked}  suppressed={total_suppressed}")

    with open(OUT_DIR / "per_file_metrics.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(
            [["file", "TP", "FP", "FN", "Precision", "Recall", "F1", "checked", "suppressed"]] + per_file_rows
        )

    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
