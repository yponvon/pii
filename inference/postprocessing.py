"""
postprocessing.py

Everything that runs on the model's raw output. Two halves:

  1. PRECISION FILTERS (drop false positives) — per-label shape/context checks,
     each diagnosed from a specific false-positive cluster. `passes_validity()`
     is the single per-detection gate; the context-gated address/unit/block/
     postal filters are applied separately in the pipeline because they also
     need span positions and the surrounding text.

  2. RECALL BOOSTERS (add missed detections) — three deterministic, context-
     gated regex boosters (postal, NRIC, email) plus value propagation, which
     re-applies an already-confirmed value to its other occurrences (the
     read-backs the model reports only once). Boosters reuse the filter regexes
     and gates above so a boosted span faces the same checks a fresh detection
     would.

Imported by inference/pipeline.py, which sequences filters then boosters.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import _FILTER_LABEL  # noqa: E402


# -- generic content-shape checks -------------------------------------------
# NRIC/FIN format: [STFGM] + 7 digits + 1 letter, e.g. S1234567A
_NRIC_STRICT_RE = re.compile(r'^[STFGMstfgm]\d{7}[A-Za-z]$')


def _is_valid_nric(text):
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
_POSTAL_KEYWORD_RE = re.compile(r'\b(?:postal|poster|post)\s+code\b', re.IGNORECASE)

_DIGIT_RE = re.compile(r'\d')


def _looks_like_postal_code(text):
    """Postal code predictions must contain at least 4 digit characters.
    Catches 'postal code' (0 digits), '119'/'010' (3 digits) without filtering
    partial predictions like '01208' (5 digits) that may TP-match address units."""
    return len(_DIGIT_RE.findall(text)) >= 4


def _looks_like_phone(text):
    """Phone number predictions must contain at least 1 digit.
    Catches names like 'Linda' or 'Ahmad Bin Muhammad' (0 digits)."""
    return bool(_DIGIT_RE.search(text))


def _looks_like_unit(text):
    """Unit/block numbers must contain at least 1 digit.
    Catches English words like 'four', 'three', 'block number' (0 digits)."""
    return bool(_DIGIT_RE.search(text))


_PHONE_LABELS  = {"SG_PHONE_NUMBER", "SG_CONTACT_NUMBER"}
_POSTAL_LABELS = {"SG_POSTAL_CODE"}
_UNIT_LABELS   = {"SG_ADDRESS_UNIT_NUMBER", "SG_ADDRESS_BLOCK_NUMBER"}


def _passes_content_filter(text, label):
    """Return False for model predictions that fail basic content sanity checks."""
    if label in _PHONE_LABELS  and not _looks_like_phone(text):
        return False
    if label in _POSTAL_LABELS and not _looks_like_postal_code(text):
        return False
    if label in _UNIT_LABELS   and not _looks_like_unit(text):
        return False
    return True


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


# -- SG_NRIC_FIN validity (full or fragment) --------------------------------
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
    """The single per-detection precision gate (shape/content checks).

    Context-gated address/unit/block/postal filters are applied separately in
    the pipeline because they also need span positions and surrounding text."""
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

# -- SG_ADDRESS_BLOCK precision filter --------------------------------------
# SG_ADDRESS_BLOCK reached only 0.46 precision despite perfect recall, from
# three systematic false-positive patterns: (1) hyphenated unit-number-format
# strings ("01-54") mistaken for blocks, (2) "Nth floor" phrases mistaken for
# blocks, and (3) bare digits with no nearby address context at all (leaked
# phone or bill fragments).

_FLOOR_PHRASE_RE = re.compile(r'\d+\s*(?:st|nd|rd|th)\s+floor', re.IGNORECASE)
_BLOCK_CONTEXT_RE = re.compile(r'\bbl(?:oc)?k\b', re.IGNORECASE)
_BLOCK_CONTEXT_WINDOW = 60


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




# -- SG_POSTAL_CODE recall booster ------------------------------------------

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
# by the email content filter: the pattern requires the literal word "at"
# immediately before the domain, not merely nearby.

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
