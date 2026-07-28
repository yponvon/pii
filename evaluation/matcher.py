"""Match predicted entities to gold entities by match quality.

This is the matcher the benchmark uses. It resolves ambiguous overlaps
deterministically. A naive matcher that walks predictions in list order and
assigns each to the first overlapping gold entry in the same label group has a
problem: because address subtypes (SG_ADDRESS, SG_ADDRESS_BLOCK,
SG_ADDRESS_UNIT, SG_POSTAL_CODE) share a label group for lenient scoring, a
broad prediction such as "123A Example Road" can claim a gold entry intended
for a tighter prediction such as "123A", which leaves the tighter prediction
unmatched and scores both as errors.

This matcher instead builds every valid (prediction, gold) candidate pair,
scores each pair by match quality, and assigns pairs greedily from best
quality first. Exact and tight matches are therefore locked in before
looser ones can consume their gold entries.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import same_label_group  # noqa: E402


def _clean(s: str) -> str:
    return re.sub(r"[\s.,!?;:\-]+", " ", s).strip().lower()


def match_quality(pred_text: str, gold_text: str) -> float:
    """Score how well two texts match; higher is better.

    An exact match scores 2.0. A partial containment scores the ratio of the
    shorter length to the longer length, so a ratio nearer 1.0 indicates a
    tighter partial match. Non-matches score -1.0.

    Args:
        pred_text: Predicted entity text.
        gold_text: Gold entity text.

    Returns:
        A match-quality score.
    """
    p, g = _clean(pred_text), _clean(gold_text)
    if not p or not g:
        return -1.0
    if p == g:
        return 2.0
    if p in g or g in p:
        shorter, longer = (p, g) if len(p) <= len(g) else (g, p)
        return len(shorter) / len(longer)  # a ratio nearer 1.0 is a tighter partial match
    return -1.0


def match_entities_fixed(pred, gold):
    """Match predictions to gold entries and tally scoring outcomes.

    Resolves ambiguous overlaps by match quality rather than list order.

    Args:
        pred: Sequence of (text, label) predicted entities.
        gold: Sequence of (text, label) gold entities.

    Returns:
        A tuple of (tp, fp, fn, errors_fp, errors_fn, tp_by_label).
    """
    candidates = []
    for i, (pt, pl) in enumerate(pred):
        for j, (gt, gl) in enumerate(gold):
            if not same_label_group(pl, gl):
                continue
            q = match_quality(pt, gt)
            if q > 0:
                candidates.append((q, i, j))

    candidates.sort(key=lambda c: c[0], reverse=True)

    pred_assigned = {}
    gold_assigned = {}
    for q, i, j in candidates:
        if i in pred_assigned or j in gold_assigned:
            continue
        pred_assigned[i] = j
        gold_assigned[j] = i

    tp_by_label = {}
    for i, j in pred_assigned.items():
        gl = gold[j][1]
        tp_by_label[gl] = tp_by_label.get(gl, 0) + 1

    errors_fn = [(gl, gt) for j, (gt, gl) in enumerate(gold) if j not in gold_assigned]
    errors_fp = [(pl, pt) for i, (pt, pl) in enumerate(pred) if i not in pred_assigned]

    tp = len(pred_assigned)
    fp = len(errors_fp)
    fn = len(errors_fn)
    return tp, fp, fn, errors_fp, errors_fn, tp_by_label
