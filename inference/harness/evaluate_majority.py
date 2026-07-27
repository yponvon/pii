"""
evaluate_majority.py

Evaluates baseline vs fine-tuned models on the 56-file majority test set
(test_fulltext_majority.jsonl), drawn randomly from typical calls rather than
the hard-49 curated set, to measure performance on typical-call inputs.

Usage:
  python3 evaluate_majority.py
"""

import json
import sys
from pathlib import Path

from gliner2 import GLiNER2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "presidio-gliner" / "scoring_utils"))
from eval_pipeline import match_entities, _prf  # noqa: E402

from evaluate_finetuned import (  # noqa: E402
    DATA_DIR, MODEL_PATH, BASELINE_LABELS, FINETUNED_LABELS, run_fulltext, THRESHOLDS,
)

# Canonical label names used by match_entities/same_label_group
_TRAIN_LABEL_TO_CANON = {
    "sg_phone_number": "SG_PHONE_NUMBER",
    "sg_address": "SG_ADDRESS",
    "sg_address_unit_number": "SG_ADDRESS_UNIT",
    "sg_address_block_number": "SG_ADDRESS_BLOCK",
    "sg_postal_code": "SG_POSTAL_CODE",
    "email_address": "EMAIL_ADDRESS",
    "sg_nric_fin": "SG_NRIC_FIN",
}


def load_majority_test_cases():
    cases = []
    with open(DATA_DIR / "test_fulltext_majority.jsonl") as f:
        for line in f:
            d = json.loads(line)
            full_text = d["input"]
            entities = d.get("output", {}).get("entities", {})
            gold = []
            for train_label, values in entities.items():
                canon = _TRAIN_LABEL_TO_CANON.get(train_label, train_label.upper())
                for v in values:
                    gold.append((v, canon))
            cases.append((full_text, gold))
    return cases


def evaluate(name, predict_fn, cases, labels, threshold):
    total_tp = total_fp = total_fn = 0
    for full_text, gold in cases:
        pred = predict_fn(full_text, labels, threshold)
        tp, fp, fn, _, _, _ = match_entities(pred, gold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    p, r, f = _prf(total_tp, total_fp, total_fn)
    print(f"{name:<28} thresh={threshold:.2f}  TP={total_tp:>4} FP={total_fp:>4} FN={total_fn:>4}  "
          f"P={p:.4f}  R={r:.4f}  F1={f:.4f}")


def main():
    cases = load_majority_test_cases()
    print(f"Majority test set: {len(cases)} files\n")

    baseline_model = GLiNER2.from_pretrained(MODEL_PATH)

    def baseline_predict(full_text, labels, threshold):
        return run_fulltext(baseline_model, full_text, labels, threshold)

    print("=== BASELINE (zero-shot + postprocessing) ===")
    for t in THRESHOLDS:
        evaluate("baseline", baseline_predict, cases, BASELINE_LABELS, t)
    print()

    variant_a_adapter = DATA_DIR / "lora_fulltext_output" / "best"
    if variant_a_adapter.exists():
        model_a = GLiNER2.from_pretrained(MODEL_PATH)
        model_a.load_adapter(str(variant_a_adapter))

        def variant_a_predict(full_text, labels, threshold):
            return run_fulltext(model_a, full_text, labels, threshold)

        print("=== VARIANT A (original, 36 hard-set files) ===")
        for t in THRESHOLDS:
            evaluate("variant_a_original", variant_a_predict, cases, FINETUNED_LABELS, t)
        print()

    expanded_adapter = DATA_DIR / "lora_fulltext_expanded_output" / "best"
    if expanded_adapter.exists():
        model_exp = GLiNER2.from_pretrained(MODEL_PATH)
        model_exp.load_adapter(str(expanded_adapter))

        def expanded_predict(full_text, labels, threshold):
            return run_fulltext(model_exp, full_text, labels, threshold)

        print("=== VARIANT A EXPANDED (272 files) ===")
        for t in THRESHOLDS:
            evaluate("variant_a_expanded", expanded_predict, cases, FINETUNED_LABELS, t)
        print()

    majority_adapter = DATA_DIR / "lora_fulltext_majority_output" / "best"
    if majority_adapter.exists():
        model_maj = GLiNER2.from_pretrained(MODEL_PATH)
        model_maj.load_adapter(str(majority_adapter))

        def majority_predict(full_text, labels, threshold):
            return run_fulltext(model_maj, full_text, labels, threshold)

        print("=== VARIANT A MAJORITY (309 files, representative split) ===")
        for t in THRESHOLDS:
            evaluate("variant_a_majority", majority_predict, cases, FINETUNED_LABELS, t)
        print()
    else:
        print(f"Majority adapter not found yet at {majority_adapter}, skipping.")


if __name__ == "__main__":
    main()
