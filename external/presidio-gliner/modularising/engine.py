"""
engine.py
---------
Core PII redaction engine.

Wraps PIIRedactor (rule-based) + GLiNER2Recognizer (context-aware model) and
exposes a single analyze_rows() method that runs the full windowed pipeline.
The engine is designed to be instantiated once (model loading is expensive)
and reused across many requests via the singleton in validator.py.

Dependency: redaction.py must be importable. When running from inside the
presidio-gliner repo, this is the ../redaction.py file. When deployed to the
guardrail service, ensure redaction.py is on the Python path or installed as
a package.
"""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Allow import of redaction.py from the parent directory of this package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redaction import PIIRedactor, GLiNER2Recognizer

from .config import DEFAULT_ENTITIES, DEFAULT_THRESHOLD, DEFAULT_WINDOW, MODEL_PATH


@dataclass
class RowResult:
    """Result for a single transcript row after redaction.

    Attributes:
        original: The unmodified row text.
        redacted: The row text with PII spans replaced by <ENTITY_TYPE> tags.
        spans: List of (start, end, entity_type) tuples identifying every
               detected PII span in the original row.
    """
    original: str
    redacted: str
    spans: List[Tuple[int, int, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal pipeline helpers (mirror the notebook's functions cell)
# ---------------------------------------------------------------------------

def _build_window(rows: List[str], i: int, window: int) -> Tuple[str, int, int]:
    """Combine row i with ±window neighbours into one string for analysis.

    GLiNER2 needs surrounding context to identify fragmented entities
    (e.g. "6267" alone is ambiguous; next to "My contact is 9040..." it's
    clearly the tail of a phone number).

    Returns:
        combined: Joined window string with rows separated by \\n.
        target_start: Char offset where row i begins in combined.
        target_end: Char offset where row i ends in combined.
    """
    win_start = max(0, i - window)
    win_end = min(len(rows), i + window + 1)
    window_rows = rows[win_start:win_end]

    combined = ""
    row_ranges = []
    for j, text in enumerate(window_rows):
        start = len(combined)
        combined += text
        row_ranges.append((start, len(combined)))
        if j < len(window_rows) - 1:
            combined += "\n"

    target_start, target_end = row_ranges[i - win_start]
    return combined, target_start, target_end


def _clamp_spans(
    results: list, target_start: int, target_end: int
) -> List[Tuple[int, int, str]]:
    """Keep only the portion of each detected span that falls inside row i.

    Presidio returns offsets relative to the combined window string.
    This remaps them to be relative to row i's start, discarding anything
    that belongs to a neighbouring row.

    Returns spans sorted right-to-left so apply_redactions can splice
    from the end without shifting earlier offsets.
    """
    spans = []
    for r in results:
        local_start = max(r.start, target_start) - target_start
        local_end = min(r.end, target_end) - target_start
        if local_start < local_end:
            spans.append((local_start, local_end, r.entity_type))
    return sorted(spans, key=lambda s: s[0], reverse=True)


def _apply_redactions(text: str, spans: List[Tuple[int, int, str]]) -> str:
    """Replace each span in text with <ENTITY_TYPE>, right-to-left.

    Processing right-to-left ensures each replacement does not shift the
    character offsets of spans still to be processed to the left.
    """
    for start, end, label in spans:
        text = text[:start] + f"<{label}>" + text[end:]
    return text


def _fallback_spans(
    target_text: str, results: list, combined: str
) -> List[Tuple[int, int, str]]:
    """Recover spans for entities that _clamp_spans missed.

    When an entity straddles a row boundary (e.g. a phone number detected as
    "9040\\n6267"), _clamp_spans returns an empty or partial span. This fallback
    searches the row text directly for the entity's value or its digit sequences.
    """
    spans = []
    for r in results:
        value = combined[r.start:r.end]
        if not value:
            continue

        if value in target_text:
            start = target_text.index(value)
            spans.append((start, start + len(value), r.entity_type))
            continue

        # Cross-row entity: extract digit sequences and redact from first to last match.
        digit_seqs = list(dict.fromkeys(re.findall(r'\d+', value)))
        positions = []
        for seq in digit_seqs:
            for m in re.finditer(re.escape(seq), target_text):
                positions.append((m.start(), m.end()))

        if positions:
            start = min(p[0] for p in positions)
            end = max(p[1] for p in positions)
            spans.append((start, end, r.entity_type))

    return sorted(spans, key=lambda s: s[0], reverse=True)


def _find_duplicate_spans(
    target_text: str,
    results: list,
    combined: str,
    covered_spans: List[Tuple[int, int, str]],
) -> List[Tuple[int, int, str]]:
    """Find uncovered duplicate occurrences of already-detected entity values.

    remove_overlapping_spans uses a greedy left-to-right sweep. If a large span
    (e.g. SG_ADDRESS absorbing "9123 4567") sets last_end past a second identical
    value, that second occurrence is silently dropped. This function does a second
    pass to catch any remaining uncovered occurrences of every detected value.
    """
    covered = [(s[0], s[1]) for s in covered_spans]
    extra = []
    seen = set()

    for r in results:
        value = combined[r.start:r.end].strip()
        if not value or value in seen:
            continue
        seen.add(value)

        for m in re.finditer(re.escape(value), target_text):
            already_covered = any(
                lo <= m.start() and m.end() <= hi for lo, hi in covered
            )
            if not already_covered:
                extra.append((m.start(), m.end(), r.entity_type))
                covered.append((m.start(), m.end()))

    return extra


def _remove_overlapping_spans(
    spans: List[Tuple[int, int, str]],
) -> List[Tuple[int, int, str]]:
    """Deduplicate overlapping spans with a greedy left-to-right sweep.

    When multiple recognizers detect the same region, keep the leftmost span
    and discard any subsequent span whose start falls before the previous end.
    Returns spans re-sorted right-to-left for apply_redactions.
    """
    sorted_asc = sorted(spans, key=lambda s: s[0])
    result = []
    last_end = -1
    for start, end, label in sorted_asc:
        if start >= last_end:
            result.append((start, end, label))
            last_end = end
    return sorted(result, key=lambda s: s[0], reverse=True)


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------

class PIIRedactionEngine:
    """Loads all recognizers once and exposes analyze_rows() for redaction.

    Args:
        model_path: HuggingFace model ID or local path for the GLiNER2 model.
        default_threshold: Minimum confidence score for a detected span to be kept.

    The engine initialises:
      - PIIRedactor: rule-based Presidio recognizers (regex for SG phone numbers,
        NRIC, addresses, account numbers).
      - GLiNER2Recognizer: context-aware NLP model added on top.
      - SpacyRecognizer is removed to avoid double-tagging names with GLiNER2.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        default_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.model_path = model_path
        self.default_threshold = default_threshold

        self._pii_redactor = PIIRedactor()
        gliner2 = GLiNER2Recognizer(model_path=model_path, threshold=default_threshold)
        self._pii_redactor.analyzer.registry.add_recognizer(gliner2)
        self._pii_redactor.analyzer.registry.remove_recognizer("SpacyRecognizer")

    def analyze_rows(
        self,
        rows: List[str],
        entities: Optional[List[str]] = None,
        window: int = DEFAULT_WINDOW,
        threshold: Optional[float] = None,
    ) -> List[RowResult]:
        """Run the full windowed PII detection and redaction pipeline on a list of rows.

        Two-pass design:
          Pass 1 — Analysis:
            For every row i, build a window of ±window surrounding rows and call
            analyzer.analyze() on the combined string. Both rule-based and GLiNER2
            recognizers run together so GLiNER2 receives enough context. Results
            are stored for all rows before any text is modified.

          Pass 2 — Redaction:
            For each row i, convert window-level results to row-level spans via:
              1. _clamp_spans      — remap offsets to row i's character range.
              2. _fallback_spans   — recover cross-boundary entities by text search.
              3. Neighbour fallback — check adjacent windows if own window found nothing.
              4. _find_duplicate_spans — catch uncovered 2nd/3rd occurrences.
              5. _remove_overlapping_spans — deduplicate overlapping spans.
              6. _apply_redactions — replace spans right-to-left with <ENTITY_TYPE>.

        Args:
            rows: List of transcript row strings.
            entities: Entity types to detect. Defaults to DEFAULT_ENTITIES.
            window: Number of rows to include above and below each target row.
            threshold: Confidence threshold override. Defaults to self.default_threshold.

        Returns:
            List of RowResult, one per input row, in the same order.
        """
        eff_entities = entities or DEFAULT_ENTITIES
        eff_threshold = threshold if threshold is not None else self.default_threshold

        # Pass 1: analyze every window and store results
        all_results = []
        for i in range(len(rows)):
            combined, target_start, target_end = _build_window(rows, i, window)
            results = self._pii_redactor.analyzer.analyze(
                text=combined,
                entities=eff_entities,
                language="en",
                score_threshold=eff_threshold,
            )
            all_results.append((results, combined, target_start, target_end))

        # Pass 2: clamp, fallback, dedup, apply
        row_results = []
        for i in range(len(rows)):
            results, combined, target_start, target_end = all_results[i]

            spans = _clamp_spans(results, target_start, target_end)

            if not spans:
                spans = _fallback_spans(rows[i], results, combined)

            if not spans:
                # Own window found nothing — check neighbouring windows
                for j in range(max(0, i - window), min(len(rows), i + window + 1)):
                    if j == i:
                        continue
                    nb_results, nb_combined, _, _ = all_results[j]
                    spans = _fallback_spans(rows[i], nb_results, nb_combined)
                    if spans:
                        break

            extra = _find_duplicate_spans(rows[i], results, combined, spans)
            spans = _remove_overlapping_spans(spans + extra)
            redacted = _apply_redactions(rows[i], spans)

            row_results.append(RowResult(original=rows[i], redacted=redacted, spans=spans))

        return row_results
