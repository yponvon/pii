"""
document_baseline_failures.py

Same full documentation as document_rule_based_failures.py, but for the
baseline model (zero-shot GLiNER2 + postprocessing), across all 420 files
in data_all, scored against data_all's own gold.

Usage:
  python3 document_baseline_failures.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")
from evaluate_finetuned import MODEL_PATH, BASELINE_LABELS, run_fulltext  # noqa: E402
from evaluate_majority import _TRAIN_LABEL_TO_CANON  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "presidio-gliner" / "scoring_utils"))
from eval_pipeline import texts_match, same_label_group, _prf  # noqa: E402

from gliner2 import GLiNER2

DATA_ALL_DIR = Path(__file__).resolve().parents[2] / "data" / "authentic_test"
THRESHOLD = 0.35


def gold_from_data_all(stem):
    p = DATA_ALL_DIR / f"{stem}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    entities = d.get("output", {}).get("entities", {})
    gold = []
    for train_label, values in entities.items():
        canon = _TRAIN_LABEL_TO_CANON.get(train_label, train_label.upper())
        for v in values:
            gold.append((v, canon))
    return gold, d["input"]


def match_detail(pred, gold):
    gold_matched = [False] * len(gold)
    pred_matched = [False] * len(pred)
    for i, (pt, pl) in enumerate(pred):
        for j, (gt, gl) in enumerate(gold):
            if gold_matched[j]:
                continue
            if texts_match(pt, gt) and same_label_group(pl, gl):
                gold_matched[j] = True
                pred_matched[i] = True
                break
    misses = [(gt, gl) for j, (gt, gl) in enumerate(gold) if not gold_matched[j]]
    extras = [(pt, pl) for i, (pt, pl) in enumerate(pred) if not pred_matched[i]]
    return misses, extras, gold_matched, pred_matched


def main():
    model = GLiNER2.from_pretrained(MODEL_PATH)
    all_files = sorted(p.stem for p in DATA_ALL_DIR.glob("*.json"))

    per_label_tp = defaultdict(int)
    per_label_fp = defaultdict(int)
    per_label_fn = defaultdict(int)
    all_misses = []
    all_extras = []
    skipped = 0
    total_tp = total_fp = total_fn = 0

    for stem in all_files:
        result = gold_from_data_all(stem)
        if result is None:
            skipped += 1
            continue
        gold, full_text = result
        pred = run_fulltext(model, full_text, BASELINE_LABELS, THRESHOLD)

        misses, extras, gold_matched, pred_matched = match_detail(pred, gold)

        # Per-label true positives: match each prediction to a gold entry, then
        # bucket the matches by label.
        gold_matched2 = [False] * len(gold)
        pred_matched2 = [False] * len(pred)
        for i, (pt, pl) in enumerate(pred):
            for j, (gt, gl) in enumerate(gold):
                if gold_matched2[j]:
                    continue
                if texts_match(pt, gt) and same_label_group(pl, gl):
                    gold_matched2[j] = True
                    pred_matched2[i] = True
                    per_label_tp[gl] += 1
                    break

        for gt, gl in misses:
            per_label_fn[gl] += 1
            all_misses.append((stem, gt, gl))
        for pt, pl in extras:
            per_label_fp[pl] += 1
            all_extras.append((stem, pt, pl))

        tp = sum(gold_matched2)
        fp = len(pred) - sum(pred_matched2)
        fn = len(gold) - sum(gold_matched2)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    print("=" * 90)
    print(f"OVERALL (420-file corpus, {skipped} skipped): TP={total_tp} FP={total_fp} FN={total_fn}")
    p, r, f = _prf(total_tp, total_fp, total_fn)
    print(f"P={p:.4f}  R={r:.4f}  F1={f:.4f}")
    print()

    print("=" * 90)
    print("PER-ENTITY-TYPE BREAKDOWN")
    print("=" * 90)
    all_labels = sorted(set(per_label_tp) | set(per_label_fp) | set(per_label_fn))
    print(f"{'label':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}")
    for label in all_labels:
        tp = per_label_tp[label]
        fp = per_label_fp[label]
        fn = per_label_fn[label]
        p, r, f = _prf(tp, fp, fn)
        print(f"{label:<20} {tp:>4} {fp:>4} {fn:>4} {p:>7.4f} {r:>7.4f} {f:>7.4f}")
    print()

    print("=" * 90)
    print(f"ALL FALSE NEGATIVES (missed entities) -- {len(all_misses)} total")
    print("=" * 90)
    by_label_misses = defaultdict(list)
    for stem, gt, gl in all_misses:
        by_label_misses[gl].append((stem, gt))
    for label in sorted(by_label_misses):
        items = by_label_misses[label]
        print(f"\n--- {label} ({len(items)} missed) ---")
        for stem, gt in items:
            print(f"  {gt!r:<60} -- {stem}")

    print()
    print("=" * 90)
    print(f"ALL FALSE POSITIVES (wrongly flagged) -- {len(all_extras)} total")
    print("=" * 90)
    by_label_extras = defaultdict(list)
    for stem, pt, pl in all_extras:
        by_label_extras[pl].append((stem, pt))
    for label in sorted(by_label_extras):
        items = by_label_extras[label]
        print(f"\n--- {label} ({len(items)} false positives) ---")
        for stem, pt in items:
            print(f"  {pt!r:<60} -- {stem}")


if __name__ == "__main__":
    main()
