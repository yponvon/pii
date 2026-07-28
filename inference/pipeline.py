"""
pipeline.py

The inference pipeline: turn a raw transcript into (text, label) entity
predictions. Orchestration only — the pieces live in sibling modules:

    preprocessing.py   spoken-number normalization (word -> digit)
    postprocessing.py  precision filters + recall boosters (clean up model output)
    labels.py          query-label lists, NORMALIZED_LABEL map, model path

Two entry points:
    run_fulltext()   single pass over the whole text (short inputs, rule-based baseline)
    run_windowed()   overlapping-window inference + reconciliation (the canonical path)

Full order for run_windowed(): normalize -> window -> model -> precision filters
-> recall boosters -> cross-window reconciliation -> value propagation -> map
spans back to the original text.

This module is imported by the redaction entry point (redact.py), the benchmark
(evaluation/), and the leak tests; it is not run directly. The label config
below is re-exported so callers can import everything from one place.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import (  # noqa: E402,F401  (re-exported for callers)
    MODEL_PATH, NORMALIZED_LABEL, BASELINE_LABELS, FINETUNED_LABELS,
    SYNTHETIC_EXTRA_LABELS, SYNTHETIC_LABELS,
)
from preprocessing import normalize_numbers, map_to_original  # noqa: E402
from postprocessing import (  # noqa: E402
    passes_validity, passes_block_context_filter,
    passes_unit_context_filter, passes_postal_context_filter,
    find_postal_codes_extended, find_nric_extended, find_email_extended,
    find_propagated_spans,
)

# Per-label confidence threshold overrides. The threshold sweep found that
# SG_ADDRESS_UNIT clears all three precision/recall/F1 gates (P=0.8056,
# R=0.8286, F1=0.8169) at confidence >= 0.75, whereas the shared 0.35 default
# gives higher F1 (P=0.7674, R=0.9429, F1=0.8461) but fails the P > 0.8 gate.
# SG_ADDRESS_BLOCK is held to a high 0.97 because bare block digits are a frequent
# false positive; only very confident block predictions are kept.
_LABEL_THRESHOLD_OVERRIDE = {"SG_ADDRESS_UNIT": 0.75, "SG_ADDRESS_BLOCK": 0.97}


def run_fulltext(model, full_text, labels, threshold, return_spans=False):
    # format_results=False bypasses the library's output formatter, whose entity
    # dedup collapses repeated values by text alone (ignoring position). That
    # dedup drops read-backs -- a PII value read back for confirmation is detected
    # by the model at full confidence but discarded in formatting. The raw result
    # keeps every occurrence with its own span, wrapped as {"entities": [ {label: [...] } ]}.
    result = model.extract_entities(full_text, labels, threshold=threshold, include_spans=True,
                                    include_confidence=True, format_results=False)
    entities = result["entities"][0] if result.get("entities") else {}
    raw_entities = []  # (text, canon, start, end, conf)
    for raw_label, ents in entities.items():
        canon = NORMALIZED_LABEL[raw_label]
        min_conf = _LABEL_THRESHOLD_OVERRIDE.get(canon, threshold)
        for e in ents:
            text = e["text"].strip()
            conf = e.get("confidence", 1.0)
            if not text or conf < min_conf or not passes_validity(text, canon):
                continue
            raw_entities.append((text, canon, e["start"], e["end"], conf))

    address_spans = [(s, e) for _t, c, s, e, _cf in raw_entities if c == "SG_ADDRESS"]

    entities, spans, confs = [], [], []
    for text, canon, start, end, conf in raw_entities:
        if canon == "SG_ADDRESS_BLOCK" and not passes_block_context_filter(
            text, start, end, full_text, address_spans
        ):
            continue
        if canon == "SG_ADDRESS_UNIT" and not passes_unit_context_filter(
            text, start, end, full_text, address_spans
        ):
            continue
        if canon == "SG_POSTAL_CODE" and not passes_postal_context_filter(
            text, start, end, full_text, address_spans
        ):
            continue
        entities.append((text, canon))
        spans.append((start, end, canon))
        confs.append(conf)

    out_spans = list(spans)  # (start, end, canon), aligned with entities so far
    # Regex boosters are deterministic and context-gated, so they are treated as high confidence.
    for ms, me, lbl in find_postal_codes_extended(full_text, spans):
        entities.append((full_text[ms:me], "SG_POSTAL_CODE")); out_spans.append((ms, me, "SG_POSTAL_CODE")); confs.append(1.0)
    for ms, me, lbl in find_nric_extended(full_text, spans):
        entities.append((full_text[ms:me], "SG_NRIC_FIN")); out_spans.append((ms, me, "SG_NRIC_FIN")); confs.append(1.0)
    for ms, me, lbl in find_email_extended(full_text, spans):
        entities.append((full_text[ms:me], "EMAIL_ADDRESS")); out_spans.append((ms, me, "EMAIL_ADDRESS")); confs.append(1.0)
    if return_spans:
        return [(t, c, s, e, cf) for (t, c), (s, e, _c), cf in zip(entities, out_spans, confs)]
    return entities


def run_windowed(model, full_text, labels, threshold, win_chars=1800, overlap=400, return_spans=False):
    """Run overlapping-window inference with cross-window reconciliation.

    Overlapping windows let PII past the encoder's roughly 512-token ceiling
    still be seen, combined with reliable cross-window reconciliation and
    spoken-number normalization.

    Returns [(text, canon), ...] by default; with return_spans=True returns
    [(text, canon, orig_start, orig_end, confidence), ...] in original-text
    coordinates, as used by the redaction output formatter.

    The transcript is first normalized (word-numbers to digits); all inference
    and reconciliation happen in normalized coordinates, and accepted spans are
    mapped back to the original text at the end so redactions land on the real
    words. On digit-only transcripts normalization is a no-op.

    Coverage guarantee: with overlap=400 and every PII entity far shorter than
    400 characters, each entity is fully contained in at least one window, so
    nothing is lost to a boundary cut.

    Conflict resolution: the same entity is often seen by two overlapping
    windows, and near a truncated edge a window may even give it a different
    label. Exactly one detection is kept per entity, chosen deterministically by
    (1) margin, the distance to the nearest truncating window edge, so the window
    that saw the entity most interior wins, then (2) confidence, (3) span length,
    and (4) position and label. A later candidate is dropped as a duplicate when
    it is the same entity (same-label overlap, or a different-label
    near-identical span with IoU >= 0.5); genuinely nested different-label spans
    (a block inside an address) are preserved."""
    norm_text, segments = normalize_numbers(full_text)
    n = len(norm_text)

    # 1) Collect every candidate (in normalized-absolute coords) with its margin.
    cands = []  # (abs_start, abs_end, canon, conf, margin, text)
    if n <= win_chars:
        for text, canon, s, e, conf in run_fulltext(model, norm_text, labels, threshold, return_spans=True):
            cands.append((s, e, canon, conf, float("inf"), text))
    else:
        step = max(1, win_chars - overlap)
        off = 0
        while True:
            end_off = min(off + win_chars, n)
            chunk = norm_text[off:end_off]
            left_doc_edge = off == 0
            right_doc_edge = end_off == n
            for text, canon, s, e, conf in run_fulltext(model, chunk, labels, threshold, return_spans=True):
                a, b = off + s, off + e
                left_margin = float("inf") if left_doc_edge else (a - off)
                right_margin = float("inf") if right_doc_edge else (end_off - b)
                cands.append((a, b, canon, conf, min(left_margin, right_margin), text))
            if end_off >= n:
                break
            off += step

    # 2) Accept best-first, dropping same-entity duplicates and keeping true nesting.
    cands.sort(key=lambda c: (-c[4], -c[3], -(c[1] - c[0]), c[0], c[2]))
    accepted = []  # (abs_start, abs_end, canon, conf)
    for a, b, canon, conf, margin, text in cands:
        conflict = False
        for xa, xb, xcanon, _xconf in accepted:
            ov = min(b, xb) - max(a, xa)
            if ov <= 0:
                continue  # disjoint, so genuinely different occurrences; keep both
            if xcanon == canon:
                conflict = True; break  # same label overlapping means the same entity
            iou = ov / (max(b, xb) - min(a, xa))
            if iou >= 0.5:
                conflict = True; break  # near-identical span with a disagreeing label
        if not conflict:
            accepted.append((a, b, canon, conf))

    # 2b) Value propagation: redact the other occurrences of each confirmed value
    # (read-backs the model reported only once), gated by specificity and context.
    # This runs in normalized coordinates over the whole transcript, so it also
    # links a digit-first mention to a spoken repeat and catches repeats across
    # windows.
    addr_spans = [(a, b) for a, b, c, _ in accepted if c == "SG_ADDRESS"]
    accepted.extend(find_propagated_spans(norm_text, accepted, addr_spans))

    # 3) Map accepted spans back to original coordinates and use the real text.
    accepted.sort(key=lambda c: (c[0], c[1]))
    out = []
    for a, b, canon, conf in accepted:
        o0, o1 = map_to_original(a, b, segments)
        text = full_text[o0:o1].strip()
        if not text:
            continue
        out.append((text, canon, o0, o1, conf) if return_spans else (text, canon))
    return out
