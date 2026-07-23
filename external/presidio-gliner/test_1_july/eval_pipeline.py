"""
eval_pipeline.py

Evaluates GLiNER2 full-text redaction (entire text column as one string) against
gold-standard redaction_annotated CSVs.

In-scope entity types (only these count toward TP/FP/FN):
  sg_phone_number, sg_address (incl. block/unit/postal), sg_nric_fin, email_address

Matching rules:
  TP  -- model redacted any overlapping span of the gold entity (partial counts)
  FN  -- gold entity completely unredacted (zero overlap)
  FP  -- model redaction has no corresponding gold entity

Output: per-file TP/FP/FN/P/R/F1, then overall totals.
        CSVs saved to run2_full_text_eval/.

Usage:
  /Users/yvonne/INTERN/echolens/venv/bin/python3 eval_pipeline.py
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from gliner2 import GLiNER2

# -- paths ---------------------------------------------------------------------

ORIG_DIR = Path("/Users/yvonne/INTERN/echolens/data_25_june/original_annotated")
GOLD_DIR = Path("/Users/yvonne/INTERN/echolens/data_25_june/redaction_annotated")
OUT_DIR  = Path("/Users/yvonne/INTERN/echolens/data_25_june/run2_full_text_eval")

# -- model ---------------------------------------------------------------------

MODEL_ID  = "fastino/gliner2-privacy-filter-PII-multi"
THRESHOLD = 0.35  # iter3: try lower threshold (was 0.4 at iter1)
LABELS = [
    "sg_phone_number",
    "sg_address",
    "sg_address_unit_number",
    "sg_address_block_number",
    "sg_postal_code",
    "email_address",
    "sg_nric_fin",
    "sg_contact_number",
]

model = GLiNER2.from_pretrained(MODEL_ID)

# -- in-scope gold label groups ------------------------------------------------
# All address variants collapse into one group for matching purposes.

_PHONE_GROUP   = {"SG_PHONE_NUMBER", "PHONE_NUMBER", "CONTACT_NUMBER",
                  "SG_CONTACT_NUMBER", "SG_MOBILE_NUMBER", "MOBILE_NUMBER", "CONTACT"}
_ADDRESS_GROUP = {"SG_ADDRESS", "ADDRESS", "SG_POSTAL_CODE", "POSTAL_CODE",
                  "SG_ADDRESS_UNIT", "SG_ADDRESS_BLOCK",
                  "SG_ADDRESS_UNIT_NUMBER", "SG_ADDRESS_BLOCK_NUMBER"}
_EMAIL_GROUP   = {"EMAIL_ADDRESS", "EMAIL"}
_NRIC_GROUP    = {"SG_NRIC_FIN", "NRIC_FIN", "NRIC", "SG_NRIC"}

_IN_SCOPE_GOLD = _PHONE_GROUP | _ADDRESS_GROUP | _EMAIL_GROUP | _NRIC_GROUP

_GROUPS = [_PHONE_GROUP, _ADDRESS_GROUP, _EMAIL_GROUP, _NRIC_GROUP]

_LABEL_TO_GROUP: dict[str, frozenset[str]] = {}
for _g in _GROUPS:
    _fs = frozenset(_g)
    for _lbl in _g:
        _LABEL_TO_GROUP[_lbl] = _fs


def same_label_group(a: str, b: str) -> bool:
    a, b = a.upper(), b.upper()
    return bool(
        _LABEL_TO_GROUP.get(a, frozenset({a}))
        & _LABEL_TO_GROUP.get(b, frozenset({b}))
    )


# -- text matching (partial-span) ----------------------------------------------

def _clean(s: str) -> str:
    return re.sub(r"[\s.,!?;:\-]+", " ", s).strip().lower()


def texts_match(pred: str, gold: str) -> bool:
    """TP if pred overlaps gold: one contains the other (case-insensitive, punct stripped)."""
    p, g = _clean(pred), _clean(gold)
    return bool(p) and bool(g) and (p in g or g in p)


# -- gold entity extraction ----------------------------------------------------

_PLACEHOLDER_SPLIT = re.compile(r"<([A-Z_]+)>")


def extract_gold_entities(orig_df: pd.DataFrame, gold_df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Parse gold rows to recover (original_text_value, label) pairs.
    Only returns entities whose label is in _IN_SCOPE_GOLD.

    Two cases:
      - Whole-row placeholder: gold row is entirely "<LABEL>" -> original row is the value.
      - Inline placeholders: split gold by <LABEL>, use surrounding text fragments
        to locate boundaries in the original row (avoids difflib LCS alignment issues).
    """
    entities: list[tuple[str, str]] = []
    for orig_row, gold_row in zip(
        orig_df["text"].fillna("").astype(str),
        gold_df["text"].fillna("").astype(str),
    ):
        if "<" not in gold_row:
            continue

        # Case 1: entire row is a single placeholder
        m = re.fullmatch(r"\s*<([A-Z_]+)>\s*", gold_row)
        if m:
            label = m.group(1)
            if label in _IN_SCOPE_GOLD:
                value = orig_row.strip()
                if value:
                    entities.append((value, label))
            continue

        # Case 2: inline placeholders mixed with text
        parts = _PLACEHOLDER_SPLIT.split(gold_row)
        pos = 0
        for k in range(0, len(parts) - 1, 2):
            text_before = parts[k]
            label       = parts[k + 1]

            if text_before:
                idx = orig_row.find(text_before, pos)
                if idx == -1:
                    break
                pos = idx + len(text_before)

            next_text = parts[k + 2] if k + 2 < len(parts) else ""
            if next_text:
                end_idx = orig_row.find(next_text, pos)
                value = orig_row[pos:end_idx].strip() if end_idx != -1 else orig_row[pos:].strip()
                if end_idx != -1:
                    pos = end_idx
            else:
                value = orig_row[pos:].strip()

            if value and label in _IN_SCOPE_GOLD:
                entities.append((value, label))

    return entities


# -- postprocessing helpers (iter2) --------------------------------------------

# NRIC/FIN format: [STFGM] + 7 digits + 1 letter, e.g. S1234567A
_NRIC_STRICT_RE = re.compile(r'^[STFGMstfgm]\d{7}[A-Za-z]$')

def _is_valid_nric(text: str) -> bool:
    """Return True only if text looks like a real SG NRIC/FIN (S/T/F/G/M + 7 digits + letter)."""
    cleaned = re.sub(r'[\s\-]', '', text.strip())
    return bool(_NRIC_STRICT_RE.match(cleaned))

# 6-digit postal code regex (word-boundary anchored)
_POSTAL6_RE = re.compile(r'\b(\d{6})\b')
# Address context keywords that make a nearby 6-digit number likely a postal code
_ADDR_CONTEXT_RE = re.compile(
    r'\b(?:Singapore|Blk|Block|Street|St|Road|Rd|Ave|Avenue|Drive|Dr|Lane|Crescent|Cres|'
    r'Close|Way|Place|Pl|Sector|Terrace|Walk|Park|Garden|Gardens|Hill|View|Heights|Estate|'
    r'Toa Payoh|Tampines|Jurong|Bedok|Ang Mo Kio|Woodlands|Yishun|Hougang|Sengkang|Punggol|'
    r'Bukit|Clementi|Pasir Ris|Bishan|Serangoon|Boon Lay|Geylang|Queenstown|Tanjong|'
    r'MacPherson|Kallang|Paya Lebar|Novena|Buona Vista)\b|#',
    re.IGNORECASE
)

_DIGIT_RE = re.compile(r'\d')

def _looks_like_postal_code(text: str) -> bool:
    """Postal code predictions must contain at least 4 digit characters.
    Catches 'postal code' (0 digits), '119'/'010' (3 digits) without filtering
    partial predictions like '01208' (5 digits) that may TP-match address units."""
    return len(_DIGIT_RE.findall(text)) >= 4

def _looks_like_phone(text: str) -> bool:
    """Phone number predictions must contain at least 1 digit.
    Catches names like 'Linda' or 'Ahmad Bin Muhammad' (0 digits)."""
    return bool(_DIGIT_RE.search(text))

def _looks_like_unit(text: str) -> bool:
    """Unit/block numbers must contain at least 1 digit.
    Catches English words like 'four', 'three', 'block number' (0 digits)."""
    return bool(_DIGIT_RE.search(text))

_PHONE_LABELS  = {"SG_PHONE_NUMBER", "SG_CONTACT_NUMBER"}
_POSTAL_LABELS = {"SG_POSTAL_CODE"}
_UNIT_LABELS   = {"SG_ADDRESS_UNIT_NUMBER", "SG_ADDRESS_BLOCK_NUMBER"}

def _passes_content_filter(text: str, label: str) -> bool:
    """Return False for model predictions that fail basic content sanity checks."""
    if label in _PHONE_LABELS  and not _looks_like_phone(text):
        return False
    if label in _POSTAL_LABELS and not _looks_like_postal_code(text):
        return False
    if label in _UNIT_LABELS   and not _looks_like_unit(text):
        return False
    return True

def _find_postal_codes_with_context(
    full_text: str, existing_spans: list[tuple[int, int, str]]
) -> list[tuple[int, int, str]]:
    """Find standalone 6-digit postal codes that appear near address context words.
    Requires address context within 80 chars to avoid false-positiving on account numbers.
    """
    new_spans: list[tuple[int, int, str]] = []
    # Don't re-add spans already predicted by the model
    covered = [(s, e) for s, e, _ in existing_spans]
    for m in _POSTAL6_RE.finditer(full_text):
        ms, me = m.start(), m.end()
        # Skip if already covered
        if any(s <= ms and me <= e for s, e in covered):
            continue
        window = full_text[max(0, ms - 80): me + 80]
        if _ADDR_CONTEXT_RE.search(window):
            new_spans.append((ms, me, "SG_POSTAL_CODE"))
    return new_spans


# -- model run + redaction -----------------------------------------------------

# iter1 backup: THRESHOLD=0.5, plain run_model (no postprocessing)
# iter1 result: P=0.7207, R=0.4706, F1=0.5694 (160 TP, 62 FP, 180 FN)

def run_model(full_text: str) -> tuple[list[tuple[str, str]], str]:
    """Run GLiNER2 with NRIC format filter + context-guarded postal code postprocessing."""
    result = model.extract_entities(full_text, LABELS, threshold=THRESHOLD, include_spans=True)

    entities: list[tuple[str, str]] = []
    spans: list[tuple[int, int, str]] = []

    for label, ents in result["entities"].items():
        label_key = label.upper()
        for e in ents:
            text = e["text"].strip()
            if not text:
                continue
            # iter2: reject NRIC predictions that don't look like a real NRIC/FIN
            if label_key == "SG_NRIC_FIN" and not _is_valid_nric(text):
                continue
            # iter4: reject phone/postal/unit predictions that lack expected digit content
            if not _passes_content_filter(text, label_key):
                continue
            entities.append((text, label_key))
            spans.append((e["start"], e["end"], label_key))

    # iter2: add 6-digit postal codes found near address-context keywords
    postal_new = _find_postal_codes_with_context(full_text, spans)
    for ms, me, lbl in postal_new:
        entities.append((full_text[ms:me], lbl))
        spans.append((ms, me, lbl))

    spans.sort(key=lambda s: s[0], reverse=True)
    redacted = full_text
    for start, end, label in spans:
        redacted = redacted[:start] + f"<{label}>" + redacted[end:]

    return entities, redacted


# -- matching ------------------------------------------------------------------

def match_entities(
    pred: list[tuple[str, str]],
    gold: list[tuple[str, str]],
) -> tuple[int, int, int, list, list, dict[str, int]]:
    """
    Returns tp, fp, fn, errors_fp, errors_fn, tp_by_label.
    Partial overlap (one text contains the other) counts as TP.
    """
    gold_matched = [False] * len(gold)
    pred_matched = [False] * len(pred)
    tp_by_label: dict[str, int] = defaultdict(int)

    for i, (pt, pl) in enumerate(pred):
        for j, (gt, gl) in enumerate(gold):
            if gold_matched[j]:
                continue
            if texts_match(pt, gt) and same_label_group(pl, gl):
                gold_matched[j] = True
                pred_matched[i] = True
                tp_by_label[gl] += 1
                break

    errors_fn = [(gl, gt) for j, (gt, gl) in enumerate(gold) if not gold_matched[j]]
    errors_fp = [(pl, pt) for i, (pt, pl) in enumerate(pred) if not pred_matched[i]]

    tp = int(sum(tp_by_label.values()))
    fp = len(errors_fp)
    fn = len(errors_fn)
    return tp, fp, fn, errors_fp, errors_fn, dict(tp_by_label)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    r = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    f = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
    return p, r, f


# -- main ----------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(ORIG_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} files\n")
    print(f"{'File':<55} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 85)

    all_fn:    list[list] = []
    all_fp:    list[list] = []
    per_file_rows: list[list] = []

    entity_tp: dict[str, int] = defaultdict(int)
    entity_fp: dict[str, int] = defaultdict(int)
    entity_fn: dict[str, int] = defaultdict(int)

    total_tp = total_fp = total_fn = 0

    for orig_path in csv_files:
        fname     = orig_path.name
        gold_path = GOLD_DIR / fname

        orig_df = pd.read_csv(orig_path)
        gold_df = pd.read_csv(gold_path)

        full_text = " ".join(orig_df["text"].fillna("").astype(str).tolist())
        pred, _   = run_model(full_text)
        gold      = extract_gold_entities(orig_df, gold_df)

        tp, fp, fn, errors_fp, errors_fn, tp_by_label = match_entities(pred, gold)
        p, r, f = _prf(tp, fp, fn)

        print(f"{fname:<55} {tp:>4} {fp:>4} {fn:>4} {p:>6.4f} {r:>6.4f} {f:>6.4f}")

        per_file_rows.append([fname, tp, fp, fn, p, r, f])
        total_tp += tp; total_fp += fp; total_fn += fn

        for gl, gt in errors_fn:
            all_fn.append([fname, gl, gt])
            entity_fn[gl] += 1

        for pl, pt in errors_fp:
            all_fp.append([fname, pl, pt])
            entity_fp[pl] += 1

        for label, count in tp_by_label.items():
            entity_tp[label] += count

    # -- overall ---------------------------------------------------------------

    p_all, r_all, f_all = _prf(total_tp, total_fp, total_fn)
    print("-" * 85)
    print(f"{'OVERALL':<55} {total_tp:>4} {total_fp:>4} {total_fn:>4} {p_all:>6.4f} {r_all:>6.4f} {f_all:>6.4f}")
    print(f"\nPrecision={p_all}  Recall={r_all}  F1={f_all}")

    # -- write CSVs ------------------------------------------------------------

    def write_csv(path: Path, header: list, rows: list[list]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([header] + rows)

    write_csv(OUT_DIR / "per_file_metrics.csv",
              ["file", "TP", "FP", "FN", "Precision", "Recall", "F1"],
              per_file_rows)

    all_labels = sorted(set(entity_tp) | set(entity_fp) | set(entity_fn))
    entity_rows: list[list] = []
    for lbl in all_labels:
        tp = entity_tp.get(lbl, 0)
        fp = entity_fp.get(lbl, 0)
        fn = entity_fn.get(lbl, 0)
        p, r, f = _prf(tp, fp, fn)
        entity_rows.append([lbl, tp, fp, fn, p, r, f])

    write_csv(OUT_DIR / "per_entity_metrics.csv",
              ["Entity", "TP", "FP", "FN", "Precision", "Recall", "F1"],
              entity_rows)

    write_csv(OUT_DIR / "errors_fn_missed.csv",
              ["file", "gold_type", "missed_value"], all_fn)

    write_csv(OUT_DIR / "errors_fp_over_redacted.csv",
              ["file", "pred_type", "over_redacted_value"], all_fp)

    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
