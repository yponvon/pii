"""
pii_redaction.py
----------------
PII redaction guardrail — drop this file into guardrail_services/ to integrate.

--- CHANGES NEEDED IN api.py ---

1. Add import:

    from guardrail_services.pii_redaction import (
        init_pii_redaction_guard,
        PIIRedactionValidator,
        _VALIDATOR as PII_VALIDATOR,
    )

2. Add request model (alongside the other Request models):

    class PIIRedactionRequest(BaseModel):
        text: str
        threshold: Optional[float] = None
        window: Optional[int] = None
        entities: Optional[List[str]] = None
        include_redacted_text: Optional[bool] = True
        include_entity_breakdown: Optional[bool] = True

3. Add inside lifespan() (alongside the other init calls):

    try:
        init_pii_redaction_guard()
    except Exception as e:
        logger.error(f"Failed to initialize PIIRedaction: {e}")

4. Add endpoint (alongside the other routes):

    @app.post("/guardrails/pii-redaction", response_model=GuardrailResult)
    async def validate_pii_redaction(request: PIIRedactionRequest):
        from guardrail_services.pii_redaction import _VALIDATOR
        try:
            if _VALIDATOR is None:
                raise HTTPException(status_code=503, detail="PIIRedaction validator not initialized")
            return _VALIDATOR.validate(
                text=request.text,
                threshold=request.threshold,
                window=request.window,
                entities=request.entities,
                include_redacted_text=request.include_redacted_text,
                include_entity_breakdown=request.include_entity_breakdown,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"validate_pii_redaction error: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

--- ALSO NEEDED ---

redaction.py must be importable (it contains PIIRedactor, SingaporeAddressRecognizer,
GLiNER2Recognizer). Copy it alongside this file or install it as a package.

--- EXAMPLE USAGE ---

    import requests

    BASE_URL = "https://chatbot-guardrails-dev.qa"
    url = f"{BASE_URL}/guardrails/pii-redaction"

    payload = {
        "text": "Hi John, you can reach me at 91234567 or john@email.com.",
        "threshold": 0.5,
        "window": 2,
        "include_redacted_text": True,
        "include_entity_breakdown": True,
    }

    response = requests.post(url, json=payload, timeout=15)
    response.json()
    # {
    #   "validator": "pii_redaction",
    #   "passed": false,
    #   "message": "Fail",
    #   "score": null,
    #   "threshold": 0.5,
    #   "version": "v1",
    #   "details": {
    #     "reason": "PII detected: EMAIL_ADDRESS, PERSON, SG_PHONE_NUMBER.",
    #     "entity_count": 3,
    #     "entities_detected": ["EMAIL_ADDRESS", "PERSON", "SG_PHONE_NUMBER"],
    #     "entity_breakdown": {"EMAIL_ADDRESS": 1, "PERSON": 1, "SG_PHONE_NUMBER": 1},
    #     "redacted_text": "Hi <PERSON>, you can reach me at <SG_PHONE_NUMBER> or <EMAIL_ADDRESS>.",
    #     "model_path": "fastino/gliner2-privacy-filter-PII-multi",
    #     "window": 2,
    #     "rows_processed": 1
    #   }
    # }
"""

from __future__ import annotations

import io
import os
import re
import sys
import threading
from collections import Counter
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from dlogger import DLogger
    logger = DLogger(config="/apps/config/guardrails-api-settings.yaml")
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from guardrail_services.validation_class import GuardrailResult
    from guardrail_services.timeout_utils import ProcessPoolManager, with_timeout
    _HAS_SERVICE_DEPS = True
except ImportError:
    _HAS_SERVICE_DEPS = False
    from pydantic import BaseModel, Field as PydanticField

    class GuardrailResult(BaseModel):  # type: ignore[no-redef]
        validator: str
        passed: bool
        message: str = ""
        score: Optional[float] = None
        threshold: Optional[float] = None
        version: Optional[str] = None
        details: Dict[str, Any] = PydanticField(default_factory=dict)

# redaction.py must be on the Python path (copy it alongside this file in the service).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from redaction import PIIRedactor, GLiNER2Recognizer


# ---------------------------------------------------------------------------
# Config — change defaults here; all can be overridden per request
# ---------------------------------------------------------------------------

MODEL_PATH: str = "fastino/gliner2-privacy-filter-PII-multi"
VERSION: str = "v1"

DEFAULT_ENTITIES: List[str] = [
    "EMAIL_ADDRESS",
    "SG_NRIC_FIN",
    "SG_PHONE_NUMBER",
    "SG_ADDRESS",
    "SG_POSTAL_CODE",
    "SG_ADDRESS_UNIT",
    "SG_ADDRESS_BLOCK",
    "PERSON",
    "ACCOUNT_NUMBER",
]

DEFAULT_THRESHOLD: float = 0.5
DEFAULT_WINDOW: int = 2
DEFAULT_MAX_TIMEOUT_S: Optional[int] = int(os.getenv("GUARDRAIL_MAX_TIMEOUT_S", 30))


# ---------------------------------------------------------------------------
# Input adapters — convert different project input types → list[str] rows
# ---------------------------------------------------------------------------

def text_to_rows(text: str) -> List[str]:
    """Split a plain text string on newlines into a list of rows.

    This is the default adapter for API calls that send text: str.
    Each newline becomes a new row passed to the engine.
    """
    return text.split("\n")


def csv_bytes_to_rows(data: bytes, text_col: str = "text") -> List[str]:
    """Parse CSV file bytes and extract the text column as rows.

    Used by batch pipelines (e.g. Azure ML) that read files as bytes.
    """
    df = pd.read_csv(io.BytesIO(data))
    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found in CSV. Available: {list(df.columns)}"
        )
    return df[text_col].fillna("").tolist()


def df_to_rows(df: pd.DataFrame, text_col: str = "text") -> List[str]:
    """Extract the text column from an existing DataFrame as rows.

    Used when the caller already holds a DataFrame in memory.
    """
    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found. Available: {list(df.columns)}"
        )
    return df[text_col].fillna("").tolist()


# ---------------------------------------------------------------------------
# Engine — core redaction pipeline (windowed analysis + span redaction)
# ---------------------------------------------------------------------------

@dataclass
class RowResult:
    """Result for a single transcript row after redaction.

    Attributes:
        original: Unmodified row text.
        redacted: Row text with PII spans replaced by <ENTITY_TYPE> tags.
        spans: List of (start, end, entity_type) for every detected span.
    """
    original: str
    redacted: str
    spans: List[Tuple[int, int, str]] = dataclass_field(default_factory=list)


def _build_window(rows: List[str], i: int, window: int) -> Tuple[str, int, int]:
    """Combine row i with ±window neighbours into one string for analysis.

    GLiNER2 needs surrounding context to identify fragmented entities
    (e.g. "6267" alone is ambiguous; next to "My contact is 9040..." it
    becomes clearly the tail of a phone number).

    Returns:
        combined: Joined window string, rows separated by \\n.
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
    This remaps them to be relative to row i, discarding anything that
    belongs to a neighbouring row. Returns spans sorted right-to-left
    so _apply_redactions can splice from the end without shifting offsets.
    """
    spans = []
    for r in results:
        local_start = max(r.start, target_start) - target_start
        local_end = min(r.end, target_end) - target_start
        if local_start < local_end:
            spans.append((local_start, local_end, r.entity_type))
    return sorted(spans, key=lambda s: s[0], reverse=True)


def _apply_redactions(text: str, spans: List[Tuple[int, int, str]]) -> str:
    """Replace each span with <ENTITY_TYPE>, right-to-left.

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

    When an entity straddles a row boundary (e.g. a number detected as
    "9040\\n6267"), _clamp_spans returns an empty or partial span. This
    fallback searches the row text directly for the entity value or its
    digit sequences.
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

    _remove_overlapping_spans uses a greedy left-to-right sweep. If a large
    span (e.g. SG_ADDRESS absorbing "9123 4567") sets last_end past a second
    identical value, that second occurrence is silently dropped. This does a
    second pass to catch any remaining uncovered occurrences.
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
    Returns spans re-sorted right-to-left for _apply_redactions.
    """
    sorted_asc = sorted(spans, key=lambda s: s[0])
    result = []
    last_end = -1
    for start, end, label in sorted_asc:
        if start >= last_end:
            result.append((start, end, label))
            last_end = end
    return sorted(result, key=lambda s: s[0], reverse=True)


class PIIRedactionEngine:
    """Loads all recognizers once and runs the windowed redaction pipeline.

    Initialises PIIRedactor (rule-based regex recognizers) and adds
    GLiNER2Recognizer (context-aware NLP model) on top. SpacyRecognizer is
    removed to avoid double-tagging names with GLiNER2.

    Args:
        model_path: HuggingFace model ID or local path for the GLiNER2 model.
        default_threshold: Minimum confidence score for a span to be kept.
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
        """Run the full windowed PII detection and redaction pipeline.

        Pass 1 — Analysis:
            For every row i, build a combined string of ±window surrounding rows
            and call analyzer.analyze(). Both rule-based and GLiNER2 recognizers
            run together so GLiNER2 gets enough context. Results are stored for
            all rows before any text is modified.

        Pass 2 — Redaction:
            For each row i:
              1. _clamp_spans        — remap window offsets to row i's range.
              2. _fallback_spans     — text search if entity crossed a row boundary.
              3. Neighbour fallback  — check adjacent windows if own window found nothing.
              4. _find_duplicate_spans — catch uncovered 2nd/3rd occurrences.
              5. _remove_overlapping_spans — deduplicate overlapping spans.
              6. _apply_redactions   — replace spans right-to-left with <ENTITY_TYPE>.

        Args:
            rows: List of transcript row strings.
            entities: Entity types to detect. Defaults to DEFAULT_ENTITIES.
            window: Rows of context above and below each target row.
            threshold: Confidence threshold override.

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


# ---------------------------------------------------------------------------
# Output formatters — callers pick what they need via include_* flags
# ---------------------------------------------------------------------------

def _format_redacted_text(row_results: List[RowResult]) -> str:
    """Join redacted rows back into a single newline-separated string."""
    return "\n".join(r.redacted for r in row_results)


def _format_entity_summary(row_results: List[RowResult]) -> Dict[str, int]:
    """Count detected entities by type across all rows.

    Returns e.g. {"SG_PHONE_NUMBER": 2, "PERSON": 1}.
    """
    counter: Counter = Counter()
    for r in row_results:
        for _, _, entity_type in r.spans:
            counter[entity_type] += 1
    return dict(counter)


def _total_entity_count(row_results: List[RowResult]) -> int:
    """Return the total number of PII spans found across all rows."""
    return sum(len(r.spans) for r in row_results)


# ---------------------------------------------------------------------------
# Singleton + worker process setup (mirrors jailbreak.py pattern)
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_VALIDATOR: Optional["PIIRedactionValidator"] = None
_WORKER_VALIDATOR: Optional["PIIRedactionValidator"] = None


def _init_worker(
    model_path: str,
    default_threshold: float,
    default_entities: List[str],
    default_window: int,
) -> None:
    """Initialize the per-process worker validator instance.

    Args:
        model_path: Model path used to load the validator.
        default_threshold: Default score threshold.
        default_entities: Default entity types to detect.
        default_window: Default window size for context rows.
    """
    global _WORKER_VALIDATOR
    _WORKER_VALIDATOR = PIIRedactionValidator(
        model_path=model_path,
        default_threshold=default_threshold,
        default_entities=default_entities,
        default_window=default_window,
    )


def _worker_validate(*args, **kwargs) -> GuardrailResult:
    """Validate input text in a worker process using the initialized validator."""
    if _WORKER_VALIDATOR is None:
        raise RuntimeError("PII redaction worker not initialized.")
    return _WORKER_VALIDATOR._validate_impl(*args, **kwargs)


if _HAS_SERVICE_DEPS:
    _POOL = ProcessPoolManager(_init_worker)


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------

class PIIRedactionValidator:
    """
    PII redaction validator combining rule-based Presidio and GLiNER2.

    Args:
        model_path: HuggingFace model ID or local path for the GLiNER2 model.
        default_entities: Entity types to detect when the caller does not specify.
        default_threshold: Minimum confidence score for a detected span to be kept.
        default_window: Number of rows above/below each target row for GLiNER2 context.

    Attributes:
        validator_name: Stable guardrail identifier ("pii_redaction").
        version: Guardrail version string for traceability.
        engine: Loaded PIIRedactionEngine (model loaded once on init).
    """

    validator_name: str = "pii_redaction"
    version: str = VERSION

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        default_entities: Optional[List[str]] = None,
        default_threshold: float = DEFAULT_THRESHOLD,
        default_window: int = DEFAULT_WINDOW,
    ) -> None:
        self.model_path = model_path
        self.default_entities = default_entities or DEFAULT_ENTITIES
        self.default_threshold = float(default_threshold)
        self.default_window = int(default_window)
        self.engine = PIIRedactionEngine(
            model_path=model_path,
            default_threshold=default_threshold,
        )

    # Uncomment @with_timeout when deploying inside guardrail_services to enable
    # per-request timeout and process-pool isolation.
    # Also change: _validate_impl = validate  →  _validate_impl = validate.__wrapped__
    #
    # @with_timeout(
    #     seconds=DEFAULT_MAX_TIMEOUT_S,
    #     pool_manager=_POOL,
    #     config_factory=lambda self: (
    #         self.model_path,
    #         float(self.default_threshold),
    #         tuple(sorted(self.default_entities)),  # hashable for pool reuse check
    #         int(self.default_window),
    #     ),
    #     initargs_factory=lambda self: (
    #         self.model_path,
    #         float(self.default_threshold),
    #         list(self.default_entities),            # passed as-is to _init_worker
    #         int(self.default_window),
    #     ),
    #     worker_func=_worker_validate,
    # )
    def validate(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        window: Optional[int] = None,
        include_redacted_text: bool = True,
        include_entity_breakdown: bool = True,
    ) -> GuardrailResult:
        """
        Detect and redact PII from a plain text string.

        Text is split on newlines into rows, analysed with a sliding window of
        ±window neighbouring rows for GLiNER2 context, then redacted row by row.

        Args:
            text: Input text. Newline-separated rows are treated as individual
                transcript turns and analysed with surrounding context.
            entities: Entity types to detect. Defaults to DEFAULT_ENTITIES.
                Pass a subset to restrict, or a superset to add custom types.
            threshold: Per-call confidence threshold override (0–1).
            window: Per-call context window size override (rows above/below).
            include_redacted_text: Include redacted_text in details. Set False
                if only metadata is needed.
            include_entity_breakdown: Include per-type entity counts in details.

        Returns:
            GuardrailResult with:
              - validator: "pii_redaction"
              - passed: True if no PII was detected
              - message: "Pass" or "Fail"
              - score: None (no single score applies to PII redaction)
              - threshold: effective threshold used
              - version: guardrail version
              - details: entity_count, entities_detected, model_path, window,
                         rows_processed, and optionally redacted_text and
                         entity_breakdown

        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("PIIRedactionValidator.validate expects a string.")

        eff_entities = entities or self.default_entities
        eff_threshold = self.default_threshold if threshold is None else float(threshold)
        eff_window = self.default_window if window is None else int(window)

        rows = text_to_rows(text)
        row_results = self.engine.analyze_rows(
            rows=rows,
            entities=eff_entities,
            window=eff_window,
            threshold=eff_threshold,
        )

        count = _total_entity_count(row_results)
        passed = count == 0
        breakdown = _format_entity_summary(row_results)

        reason = (
            f"PII detected: {', '.join(sorted(breakdown.keys()))}."
            if not passed
            else "No PII detected."
        )

        details: Dict[str, Any] = {
            "reason": reason,
            "entity_count": count,
            "entities_detected": sorted(breakdown.keys()),
            "model_path": self.model_path,
            "window": eff_window,
            "rows_processed": len(rows),
        }
        if include_redacted_text:
            details["redacted_text"] = _format_redacted_text(row_results)
        if include_entity_breakdown:
            details["entity_breakdown"] = breakdown

        return GuardrailResult(
            validator=self.validator_name,
            passed=passed,
            message="Pass" if passed else "Fail",
            score=None,
            threshold=eff_threshold,
            version=self.version,
            details=details,
        )

    # Undecorated validate for worker process execution.
    # When @with_timeout is uncommented above, change to: _validate_impl = validate.__wrapped__
    _validate_impl = validate


# ---------------------------------------------------------------------------
# Startup initializer (called in api.py lifespan)
# ---------------------------------------------------------------------------

def init_pii_redaction_guard(
    model_path: Optional[str] = None,
    default_entities: Optional[List[str]] = None,
    default_threshold: float = DEFAULT_THRESHOLD,
    default_window: int = DEFAULT_WINDOW,
) -> None:
    """Initialize the PII redaction validator singleton at API startup.

    Called once during the FastAPI lifespan. Subsequent calls with the same
    configuration are no-ops. Calling with a different configuration raises
    RuntimeError — restart the service to change the model or defaults.

    Args:
        model_path: Model path override. Falls back to PII_MODEL_PATH env var,
            then to the default HuggingFace model ID.
        default_entities: Entity types to detect by default.
        default_threshold: Default confidence threshold. Defaults to 0.5.
        default_window: Default context window size. Defaults to 2.
    """
    global _VALIDATOR

    resolved_model_path = (
        model_path
        or os.getenv("PII_MODEL_PATH")
        or MODEL_PATH
    )

    with _LOCK:
        if _VALIDATOR is not None:
            if (
                _VALIDATOR.model_path != resolved_model_path
                or _VALIDATOR.default_threshold != float(default_threshold)
                or _VALIDATOR.default_window != int(default_window)
            ):
                raise RuntimeError(
                    "PIIRedactionGuard already initialized with a different configuration. "
                    "Restart the service to change model_path/threshold/window."
                )
            return

        _VALIDATOR = PIIRedactionValidator(
            model_path=resolved_model_path,
            default_entities=default_entities,
            default_threshold=default_threshold,
            default_window=default_window,
        )
