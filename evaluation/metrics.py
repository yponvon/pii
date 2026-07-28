"""
metrics.py

Scoring primitives used by the benchmark and the matcher:

  - same_label_group -- label-group equality for lenient matching (all address
                        subtypes collapse into one group, phone synonyms into
                        another, and so on)
  - _prf             -- precision / recall / F1 from TP/FP/FN counts

Pure functions, no model or data dependency. The entity matcher that uses
same_label_group lives in matcher.py; the per-label content filters live in
inference/filters.py.
"""

from __future__ import annotations


# -- in-scope label groups -----------------------------------------------------
# All address variants collapse into one group for lenient matching purposes.

_PHONE_GROUP   = {"SG_PHONE_NUMBER", "PHONE_NUMBER", "CONTACT_NUMBER",
                  "SG_CONTACT_NUMBER", "SG_MOBILE_NUMBER", "MOBILE_NUMBER", "CONTACT"}
_ADDRESS_GROUP = {"SG_ADDRESS", "ADDRESS", "SG_POSTAL_CODE", "POSTAL_CODE",
                  "SG_ADDRESS_UNIT", "SG_ADDRESS_BLOCK",
                  "SG_ADDRESS_UNIT_NUMBER", "SG_ADDRESS_BLOCK_NUMBER"}
_EMAIL_GROUP   = {"EMAIL_ADDRESS", "EMAIL"}
_NRIC_GROUP    = {"SG_NRIC_FIN", "NRIC_FIN", "NRIC", "SG_NRIC"}

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


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    r = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    f = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
    return p, r, f
