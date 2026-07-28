"""
Inference pipeline for the fine-tuned PII model.

Turns a raw transcript into (text, label) entity predictions through the
production path: spoken-number normalization, overlapping-window inference,
per-label precision filters, and regex recall boosters. The two entry points
are run_windowed() (the canonical path, character-overlap windows) and
run_fulltext() (single-pass, used for short text and the rule-based baseline).

This module is imported by the benchmark (run_frozen_comparison.py,
benchmark_all_labels.py), the redaction entry point (redact_output.py), and the
leak tests; it is not run directly.
"""

import re
import sys
from pathlib import Path

_PII_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PII_ROOT / "utils"))
from scoring import (  # noqa: E402
    _is_valid_nric, _passes_content_filter, _ADDR_CONTEXT_RE, _POSTAL6_RE,
)

MODEL_PATH = "fastino/gliner2-privacy-filter-PII-multi"

# -- labels ---------------------------------------------------------------
# The zero-shot baseline needs sg_contact_number queried as a synonym of
# sg_phone_number. The fine-tuned models were trained with sg_contact_number
# already merged into sg_phone_number, so they are queried with the 7 base
# labels only.

BASELINE_LABELS = ["sg_phone_number", "sg_contact_number", "sg_address", "sg_address_unit_number",
                   "sg_address_block_number", "sg_postal_code", "email_address", "sg_nric_fin"]
FINETUNED_LABELS = ["sg_phone_number", "sg_address", "sg_address_unit_number",
                    "sg_address_block_number", "sg_postal_code", "email_address", "sg_nric_fin"]

# The synthetic corpus (generated_data_14jul/) adds two labels on top of the
# base 7. These are kept in a separate list so callers that import
# FINETUNED_LABELS are unaffected; only models trained on the synthetic split
# should be queried with SYNTHETIC_LABELS.
SYNTHETIC_EXTRA_LABELS = ["account_number", "full_name"]
SYNTHETIC_LABELS = FINETUNED_LABELS + SYNTHETIC_EXTRA_LABELS

CANON = {
    "sg_phone_number": "SG_PHONE_NUMBER", "sg_contact_number": "SG_PHONE_NUMBER",
    "sg_address": "SG_ADDRESS", "sg_address_unit_number": "SG_ADDRESS_UNIT",
    "sg_address_block_number": "SG_ADDRESS_BLOCK", "sg_postal_code": "SG_POSTAL_CODE",
    "email_address": "EMAIL_ADDRESS", "sg_nric_fin": "SG_NRIC_FIN",
    # Synthetic-corpus labels; additive and harmless for models that never emit them.
    "account_number": "ACCOUNT_NUMBER", "full_name": "FULL_NAME",
}
_FILTER_LABEL = {
    "SG_PHONE_NUMBER": "SG_PHONE_NUMBER", "SG_ADDRESS": "SG_ADDRESS",
    "SG_ADDRESS_UNIT": "SG_ADDRESS_UNIT_NUMBER", "SG_ADDRESS_BLOCK": "SG_ADDRESS_BLOCK_NUMBER",
    "SG_POSTAL_CODE": "SG_POSTAL_CODE", "EMAIL_ADDRESS": "EMAIL_ADDRESS", "SG_NRIC_FIN": "SG_NRIC_FIN",
    # No format filter for ACCOUNT_NUMBER (per spec, every grouping counts) and
    # none for FULL_NAME. _passes_content_filter only gates the phone, postal,
    # and unit labels, so these pass through untouched. The mapping exists purely
    # so passes_validity() does not raise a KeyError when the model emits them.
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER", "FULL_NAME": "FULL_NAME",
}

_POSTAL_KEYWORD_RE = re.compile(r'\b(?:postal|poster|post)\s+code\b', re.IGNORECASE)


def _slice_of_longer_number(full_text, ms, me, reach=3):
    """Report whether a 6-digit run is embedded in a longer segmented number.

    Returns True when another digit group sits within `reach` characters on
    either side (for example the '372717' inside '896 372717 1'). That is the
    signature of a slice of a longer number, typically an account or reference
    number, rather than a standalone Singapore postal code, which is normally
    preceded by 'Singapore' or an area name and not by bare digits. This guards
    the postal booster against redacting account-number digits."""
    before = full_text[max(0, ms - reach):ms]
    after = full_text[me:me + reach]
    return any(c.isdigit() for c in before) or any(c.isdigit() for c in after)


def find_postal_codes_extended(full_text, existing_spans):
    covered = [(s, e) for s, e, _ in existing_spans]
    new_spans = []
    for m in _POSTAL6_RE.finditer(full_text):
        ms, me = m.start(), m.end()
        if any(s <= ms and me <= e for s, e in covered):
            continue
        if _slice_of_longer_number(full_text, ms, me):
            continue  # part of a longer number such as an account number, not a postal code
        window = full_text[max(0, ms - 80): me + 80]
        if _ADDR_CONTEXT_RE.search(window) or _POSTAL_KEYWORD_RE.search(window):
            new_spans.append((ms, me, "SG_POSTAL_CODE"))
    return new_spans


# -- EMAIL_ADDRESS content filter -------------------------------------------
# Unlike the phone, postal, and unit labels, EMAIL_ADDRESS had no content-shape
# validation, which produced two systematic false-positive patterns:
# (1) shapeless unrelated content ("australian residences", bare numbers,
# hyphenated phone-like strings), and (2) the utility company's own domain
# ("SP Group") mis-transcribed as sbgroup/htgroup/spgroup and flagged as a
# personal customer email. This filter requires an email-like shape and denies
# the company domain.

_EMAIL_SHAPE_RE = re.compile(
    r'@|\bat\b|\.(?:com|sg|net|org|edu|gov)\b|\bdot\s+(?:com|sg|net|org)\b',
    re.IGNORECASE,
)
_COMPANY_DOMAIN_DENYLIST_RE = re.compile(
    r'\b(?:s[bp]|ht)\s*-?\s*group\b', re.IGNORECASE,
)


def _passes_email_filter(text):
    if not _EMAIL_SHAPE_RE.search(text):
        return False
    if _COMPANY_DOMAIN_DENYLIST_RE.search(text):
        return False
    return True


# -- SG_PHONE_NUMBER precision filter ---------------------------------------
# Phone false positives cluster into two shapes: (1) monetary amounts
# ('22289.53', '$4 plus'), and (2) letter-plus-digit combinations that match
# the NRIC or unit format instead ('502P', 'A746'). This filter rejects both.
# Misses are mostly short digit-group fragments (a phone read out in chunks),
# which is a recall issue that a precision filter cannot address without
# harming genuine short-fragment catches, so it is left alone here.

_MONEY_RE = re.compile(r'\$|\d\.\d{2}\b')
_LETTER_DIGIT_FORMAT_RE = re.compile(r'^[A-Za-z]\d+$|^\d+[A-Za-z]$')


def _passes_phone_filter(text):
    if _MONEY_RE.search(text):
        return False
    if _LETTER_DIGIT_FORMAT_RE.match(text.strip()):
        return False
    return True


# -- SG_POSTAL_CODE precision filter ----------------------------------------
# The upstream format check only requires four or more digits, which is far
# too loose: real Singapore postal codes are exactly 6 digits with no hyphens.
# Half of the observed false positives did not even match that format ('0887'
# has 4 digits, '13452' and '07502' have 5, plus two hyphenated fragments).
# The rest were clean 6-digit numbers that need address context to be told
# apart from coincidental 6-digit numbers, the same problem the block and unit
# labels face.

_POSTAL_CONTEXT_WINDOW = 150


def passes_postal_context_filter(text, start, end, full_text, address_spans):
    digits_only = re.sub(r'[\s\-]', '', text.strip())
    if not re.fullmatch(r'\d{6}', digits_only):
        return False
    window = full_text[max(0, start - _POSTAL_CONTEXT_WINDOW): end + _POSTAL_CONTEXT_WINDOW]
    if _ADDR_CONTEXT_RE.search(window) or _POSTAL_KEYWORD_RE.search(window):
        return True
    for a_start, a_end in address_spans:
        if abs(a_start - end) <= _POSTAL_CONTEXT_WINDOW or abs(start - a_end) <= _POSTAL_CONTEXT_WINDOW:
            return True
    return False


# Match an NRIC or FIN, whole or in pieces. The strict _is_valid_nric only
# accepts a full S/T/F/G/M prefix plus 7 digits plus a letter, which silently
# dropped every fragment of an NRIC dictated in pieces (for example 'S117' plus
# '1-2-3-4-A' making S9876123A), leaking a full NRIC even though the fine-tuned
# model emitted those fragments at confidence 1.0. We therefore also accept NRIC
# fragments: an S/T/F/G/M-prefixed run, or a digits-plus-letter tail such as
# '5842H' or '840D'. The trade-off is that harmless last-4 verification mentions
# get over-redacted, which is acceptable: over-redacting a non-identifying
# fragment is far cheaper than leaking a full NRIC.
_NRIC_FRAGMENT_PREFIX_RE = re.compile(r'^[STFGM]\d{1,7}[A-Z]?$', re.IGNORECASE)
_NRIC_FRAGMENT_TAIL_RE = re.compile(r'^\d{2,6}[A-Z]$', re.IGNORECASE)


def _is_nric_like(text):
    """Return True for a full NRIC or a fragment of one; see the note above."""
    cleaned = re.sub(r'[\s\-]', '', text.strip()).upper()
    return bool(_is_valid_nric(text)
                or _NRIC_FRAGMENT_PREFIX_RE.match(cleaned)
                or _NRIC_FRAGMENT_TAIL_RE.match(cleaned))


def passes_validity(text, canon_label):
    filter_label = _FILTER_LABEL[canon_label]
    if canon_label == "SG_NRIC_FIN" and not _is_nric_like(text):
        return False
    if canon_label == "EMAIL_ADDRESS" and not _passes_email_filter(text):
        return False
    if canon_label == "SG_PHONE_NUMBER" and not _passes_phone_filter(text):
        return False
    return _passes_content_filter(text, filter_label)


# -- SG_ADDRESS_UNIT precision filter ---------------------------------------
# Unit false positives were mostly bare digit fragments with no unit-number
# shape or context ('0333', '1704', '8000'), plus clearly unrelated content:
# 'outbound 123' (a call-type label), '1-800' (a toll-free phone prefix), and
# '1-1-5-2-6' / '8-5-7-2-2' (digit-by-digit dictated phone numbers). Real
# Singapore unit numbers are typically a clean "##-####" hyphenated format, so
# that shape is trusted outright; otherwise a nearby unit/# keyword or
# SG_ADDRESS proximity is required.

_UNIT_HYPHEN_FORMAT_RE = re.compile(r'^#?\d{2}-\d{2,4}$')
_UNIT_CONTEXT_RE = re.compile(r'\bunit\b|#', re.IGNORECASE)
_UNIT_NEGATIVE_CONTEXT_RE = re.compile(
    r'\b(?:reference|account)\s+number\b|\boutbound\b|\binbound\b', re.IGNORECASE
)
_UNIT_CONTEXT_WINDOW = 60
_NEGATIVE_CONTEXT_WINDOW = 100  # wider window for exclusion checks only; lower risk than widening inclusion


def passes_unit_context_filter(text, start, end, full_text, address_spans):
    if _UNIT_HYPHEN_FORMAT_RE.match(text.strip()):
        return True
    window = full_text[max(0, start - _UNIT_CONTEXT_WINDOW): end + _UNIT_CONTEXT_WINDOW]
    neg_window = full_text[max(0, start - _NEGATIVE_CONTEXT_WINDOW): end + _NEGATIVE_CONTEXT_WINDOW]
    if _FLOOR_PHRASE_RE.search(window) or _UNIT_NEGATIVE_CONTEXT_RE.search(neg_window):
        return False
    if _UNIT_CONTEXT_RE.search(window):
        return True
    for a_start, a_end in address_spans:
        if abs(a_start - end) <= _UNIT_CONTEXT_WINDOW or abs(start - a_end) <= _UNIT_CONTEXT_WINDOW:
            return True
    return False


# -- SG_ADDRESS_BLOCK precision filter --------------------------------------
# SG_ADDRESS_BLOCK reached only 0.46 precision despite perfect recall, from
# three systematic false-positive patterns: (1) hyphenated unit-number-format
# strings ("01-54") mistaken for blocks, (2) "Nth floor" phrases mistaken for
# blocks, and (3) bare digits with no nearby address context at all (leaked
# phone or bill fragments).

_FLOOR_PHRASE_RE = re.compile(r'\d+\s*(?:st|nd|rd|th)\s+floor', re.IGNORECASE)
_BLOCK_CONTEXT_RE = re.compile(r'\bbl(?:oc)?k\b', re.IGNORECASE)
_BLOCK_CONTEXT_WINDOW = 60


def passes_block_context_filter(text, start, end, full_text, address_spans):
    """Filter SG_ADDRESS_BLOCK candidates beyond passes_validity.

    Requires no hyphen (a hyphen is the unit-number signature), no enclosing
    'Nth floor' phrase, and either a 'block'/'blk' keyword nearby or proximity
    to an already-detected SG_ADDRESS span."""
    if "-" in text:
        return False
    if _UNIT_CONTEXT_RE.search(text):
        return False  # the candidate's own text says "unit"/"#", so it is not a block

    window = full_text[max(0, start - _BLOCK_CONTEXT_WINDOW): end + _BLOCK_CONTEXT_WINDOW]
    neg_window = full_text[max(0, start - _NEGATIVE_CONTEXT_WINDOW): end + _NEGATIVE_CONTEXT_WINDOW]
    if _FLOOR_PHRASE_RE.search(window) or _UNIT_NEGATIVE_CONTEXT_RE.search(neg_window):
        return False

    if _BLOCK_CONTEXT_RE.search(window):
        return True
    for a_start, a_end in address_spans:
        if abs(a_start - end) <= _BLOCK_CONTEXT_WINDOW or abs(start - a_end) <= _BLOCK_CONTEXT_WINDOW:
            return True
    return False


# -- SG_NRIC_FIN recall booster ---------------------------------------------
# A full NRIC or FIN has a far more distinctive shape than a phone number
# (letter, 7 digits, letter, versus a phone's generic 8-digit run), so a
# context-gated regex booster is more likely to help here than it does for
# phone. A nearby "nric"/"ic"/"fin" keyword is required to avoid false
# positives.

_FULL_NRIC_BOOST_RE = re.compile(r'\b[STFG]\d{7}[A-Za-z]\b')
_NRIC_KEYWORD_RE = re.compile(r'\bnric\b|\bic\b|\bfin\b|\bidentification\b|\bidentity\b', re.IGNORECASE)
_NRIC_CONTEXT_WINDOW = 80
# "name, NRIC" adjacency (no dedicated question). A full NRIC's shape is
# already rare and specific enough that this does not need the keyword gate too.
_NAME_ADJACENT_RE = re.compile(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2},\s*$')


def find_nric_extended(full_text, existing_spans):
    covered = [(s, e) for s, e, _ in existing_spans]
    new_spans = []
    for m in _FULL_NRIC_BOOST_RE.finditer(full_text):
        ms, me = m.start(), m.end()
        if any(s <= ms and me <= e for s, e in covered):
            continue
        window = full_text[max(0, ms - _NRIC_CONTEXT_WINDOW): me + _NRIC_CONTEXT_WINDOW]
        preceding = full_text[max(0, ms - 40): ms]
        if _NRIC_KEYWORD_RE.search(window) or _NAME_ADJACENT_RE.search(preceding):
            new_spans.append((ms, me, "SG_NRIC_FIN"))
    return new_spans


# -- EMAIL_ADDRESS recall booster -------------------------------------------
# Targets the tightest, lowest-risk part of the recall gap: an "at <domain>"
# phrase in immediate adjacency. The model missed clean domain-only mentions
# such as 'at example.com' and 'At example.com.sg' even without a preceding
# spelled-out name. It is deliberately narrow (exact "at X.tld" adjacency, no
# broader keyword gating) to avoid the company-domain false positives handled
# by the email content filter above: the pattern requires the literal word
# "at" immediately before the domain, not merely nearby.

_EMAIL_DOMAIN_BOOST_RE = re.compile(
    r'\b[Aa]t\s+([\w.-]+\.(?:com|net|org|edu)(?:\.sg)?)\b'
)


def find_email_extended(full_text, existing_spans):
    covered = [(s, e) for s, e, _ in existing_spans]
    new_spans = []
    for m in _EMAIL_DOMAIN_BOOST_RE.finditer(full_text):
        ms, me = m.start(1), m.end(1)
        if any(s <= ms and me <= e for s, e in covered):
            continue
        domain_text = m.group(1)
        if _COMPANY_DOMAIN_DENYLIST_RE.search(domain_text):
            continue
        new_spans.append((ms, me, "EMAIL_ADDRESS"))
    return new_spans


# -- value-propagation recall booster ---------------------------------------
# The model reports a repeated value only once (one span per unique value), so
# read-backs, where the agent or customer repeats a number on the next turn,
# leak as plain text. The context filters cannot help: they only judge spans the
# model already emitted, and the repeats were never proposed. This booster
# proposes them. For each value the model already confirmed as PII, it finds the
# other occurrences and re-gates each through the same per-label filter a fresh
# detection would face, plus a specificity guard so an ambiguous short value
# (for example "23" tagged as a block) is never propagated across the transcript.

def _propagation_ok(value, label):
    """Guard propagation to values distinctive enough to reidentify safely.

    Only values where another exact occurrence is almost certainly the same
    identifier, not a coincidence, are propagated. Short or ambiguous labels
    (block, unit) and stray short numbers are excluded, since context alone
    cannot make "23" safe to blank everywhere."""
    v = value.strip()
    digits = re.sub(r"\D", "", v)
    alnum = re.sub(r"[^A-Za-z0-9]", "", v)
    if label in ("SG_PHONE_NUMBER", "ACCOUNT_NUMBER", "SG_POSTAL_CODE", "SG_NRIC_FIN"):
        return len(digits) >= 6          # full identifiers; excludes stray short numbers
    if label == "EMAIL_ADDRESS":
        return len(alnum) >= 6
    if label == "SG_ADDRESS":
        return len(v.split()) >= 2 and len(alnum) >= 6
    if label == "FULL_NAME":
        return len(v.split()) >= 2       # multi-token names only, to avoid common first names
    return False                         # SG_ADDRESS_BLOCK / SG_ADDRESS_UNIT are too ambiguous


def find_propagated_spans(text, accepted, address_spans):
    """Propagate accepted PII values to their other occurrences in the text.

    Args:
        text: The transcript being scanned.
        accepted: Already-accepted spans as (start, end, canon, conf) tuples in
            `text` coordinates.
        address_spans: SG_ADDRESS spans used by the postal context filter.

    Returns:
        New (start, end, canon, conf) spans for the other occurrences of each
        already-accepted value, each gated by the specificity guard and the
        label's context filter."""
    occupied = [(a, b) for a, b, _c, _cf in accepted]
    new = []
    for a, b, canon, _conf in list(accepted):
        value = text[a:b].strip()
        if not value or not _propagation_ok(value, canon):
            continue
        pat = re.compile(r'(?<![A-Za-z0-9])' + re.escape(value) + r'(?![A-Za-z0-9])')
        for m in pat.finditer(text):
            ns, ne = m.start(), m.end()
            if any(not (ne <= xs or ns >= xe) for xs, xe in occupied):
                continue  # this occurrence is already covered by an accepted span
            # Re-gate through the same filter a fresh detection at this spot would face.
            if canon == "SG_PHONE_NUMBER" and not _passes_phone_filter(value):
                continue
            if canon == "SG_POSTAL_CODE" and not passes_postal_context_filter(
                value, ns, ne, text, address_spans):
                continue
            new.append((ns, ne, canon, 1.0))
            occupied.append((ns, ne))
    return new


# -- inference + scoring ----------------------------------------------------

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
        canon = CANON[raw_label]
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


# -- spoken-number normalization (word to digit) ----------------------------
# Numbers dictated as words ("eight nine five") are nearly invisible to a model
# trained on digit patterns, causing full phone, account, and postal leaks
# (measured recall of 0.04 on spelled-out PII). Spelled-out runs are collapsed
# to digits before inference, then predicted spans are mapped back to the
# original word spans so a redaction covers the real text rather than the
# digits. The word-to-digit mapping is unambiguous ("eight" is always 8), so
# unlike an ambiguous regex it carries no rigidity risk, and on transcripts with
# no spelled-out numbers it is a no-op.

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


def _map_to_original(ns, ne, segments):
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
        o0, o1 = _map_to_original(a, b, segments)
        text = full_text[o0:o1].strip()
        if not text:
            continue
        out.append((text, canon, o0, o1, conf) if return_spans else (text, canon))
    return out
