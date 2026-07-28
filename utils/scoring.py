"""
scoring.py

Shared scoring and content-filter helpers used by the benchmark and the
inference pipeline:

  - same_label_group / _prf        -- scoring primitives (label-group equality
                                       for lenient matching, and P/R/F1)
  - _is_valid_nric                 -- NRIC/FIN format validation
  - _passes_content_filter         -- per-label shape sanity checks (a phone
                                       must have digits, a postal code >= 4, ...)
  - _ADDR_CONTEXT_RE / _POSTAL6_RE -- regexes for the postal-code recall booster

These are pure functions and regexes with no model or data-file dependencies.
The actual entity matcher lives in inference/harness/span_matcher.py
(match_entities_fixed), which is the matcher the benchmark uses.
"""

from __future__ import annotations

import re


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


# -- NRIC/FIN validation -------------------------------------------------------
# NRIC/FIN format: [STFGM] + 7 digits + 1 letter, e.g. S1234567A

_NRIC_STRICT_RE = re.compile(r'^[STFGMstfgm]\d{7}[A-Za-z]$')


def _is_valid_nric(text: str) -> bool:
    """Return True only if text looks like a real SG NRIC/FIN (S/T/F/G/M + 7 digits + letter)."""
    cleaned = re.sub(r'[\s\-]', '', text.strip())
    return bool(_NRIC_STRICT_RE.match(cleaned))


# -- postal-code booster regexes -----------------------------------------------

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


# -- per-label content filters -------------------------------------------------

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


# -- metrics -------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    r = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
    f = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
    return p, r, f
