"""
preprocessing.py

Spoken-number normalization (word -> digit), run before inference.

Numbers dictated as words ("eight nine five") are nearly invisible to a model
trained on digit patterns, causing full phone, account, and postal leaks
(measured recall of 0.04 on spelled-out PII). Spelled-out runs are collapsed to
digits before inference, then predicted spans are mapped back to the original
word spans so a redaction covers the real text rather than the digits. The
word-to-digit mapping is unambiguous ("eight" is always 8), so unlike an
ambiguous regex it carries no rigidity risk, and on transcripts with no
spelled-out numbers it is a no-op.
"""

import re

_NUM_WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "oh": "0",
}
_NUM_WORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|oh|double|triple)"
_NUM_RUN_RE = re.compile(rf"\b{_NUM_WORD}(?:[\s,\-]+{_NUM_WORD})*\b", re.IGNORECASE)
_NUM_TOKEN_RE = re.compile(_NUM_WORD, re.IGNORECASE)


def _run_to_digits(run_text):
    """Convert a spoken digit run to its digits.

    'double' and 'triple' repeat the next digit ('double one' becomes '11');
    every other word maps to a single digit."""
    tokens = _NUM_TOKEN_RE.findall(run_text)
    digits, i = [], 0
    while i < len(tokens):
        word = tokens[i].lower()
        if word in ("double", "triple"):
            repeat = 2 if word == "double" else 3
            if i + 1 < len(tokens) and tokens[i + 1].lower() in _NUM_WORD_TO_DIGIT:
                digits.append(_NUM_WORD_TO_DIGIT[tokens[i + 1].lower()] * repeat)
                i += 2
                continue
            i += 1
            continue
        if word in _NUM_WORD_TO_DIGIT:
            digits.append(_NUM_WORD_TO_DIGIT[word])
        i += 1
    return "".join(digits)


def normalize_numbers(text):
    """Normalize spelled-out numbers to digits and record the span mapping.

    Returns (normalized_text, segments), where each segment is
    (norm_start, norm_end, orig_start, orig_end, is_number). A spelled-out run is
    converted only when it yields at least 2 digits, so incidental single words
    ('one moment') are left untouched; everything else is copied
    character-for-character, so a digit-only transcript normalizes to itself."""
    segments, out, pos, npos = [], [], 0, 0
    for m in _NUM_RUN_RE.finditer(text):
        gap = text[pos:m.start()]
        out.append(gap)
        segments.append((npos, npos + len(gap), pos, m.start(), False))
        npos += len(gap)
        digits = _run_to_digits(m.group())
        if len(digits) >= 2:
            out.append(digits)
            segments.append((npos, npos + len(digits), m.start(), m.end(), True))
            npos += len(digits)
        else:
            original = text[m.start():m.end()]
            out.append(original)
            segments.append((npos, npos + len(original), m.start(), m.end(), False))
            npos += len(original)
        pos = m.end()
    tail = text[pos:]
    out.append(tail)
    segments.append((npos, npos + len(tail), pos, len(text), False))
    return "".join(out), segments


def map_to_original(ns, ne, segments):
    """Map a [ns, ne) span from normalized coordinates back to the original text.

    A number segment maps as a whole (the digit run back to the whole word run);
    other segments map character-for-character."""
    o0 = o1 = None
    for a, b, oa, ob, is_number in segments:
        if b <= ns or a >= ne:
            continue
        if is_number:
            lo, hi = oa, ob
        else:
            lo, hi = oa + (max(ns, a) - a), oa + (min(ne, b) - a)
        o0 = lo if o0 is None else min(o0, lo)
        o1 = hi if o1 is None else max(o1, hi)
    return (o0, o1) if o0 is not None else (ns, ne)
