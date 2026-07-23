"""
validator.py
------------
PIIRedactionValidator — thread-safe singleton following the guardrail_services pattern.

This file is the main entry point for the guardrail service. It mirrors the
structure of guardrail_services/jailbreak.py exactly so it can be dropped
into the service with minimal changes.

--- HOW TO INTEGRATE INTO api.py ---

1. Add import at the top of api.py:

    from guardrail_services.pii_redaction import (
        init_pii_redaction_guard,
        PIIRedactionValidator,
        _VALIDATOR as PII_VALIDATOR,
    )

2. Add request model:

    class PIIRedactionRequest(BaseModel):
        text: str
        threshold: Optional[float] = None
        window: Optional[int] = None
        entities: Optional[List[str]] = None
        include_redacted_text: Optional[bool] = True
        include_entity_breakdown: Optional[bool] = True

3. Add to lifespan():

    try:
        init_pii_redaction_guard()
    except Exception as e:
        logger.error(f"Failed to initialize PIIRedaction: {e}")

4. Add endpoint:

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
"""

import os
import threading
from typing import List, Optional

try:
    from dlogger import DLogger
    logger = DLogger(config="/apps/config/guardrails-api-settings.yaml")
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Service dependencies — available when running inside guardrail_services.
# Falls back to a local Pydantic model so this module works standalone for testing.
try:
    from guardrail_services.validation_class import GuardrailResult
    from guardrail_services.timeout_utils import ProcessPoolManager, with_timeout
    _HAS_SERVICE_DEPS = True
except ImportError:
    _HAS_SERVICE_DEPS = False
    from typing import Any, Dict
    from pydantic import BaseModel, Field

    class GuardrailResult(BaseModel):  # type: ignore[no-redef]
        validator: str
        passed: bool
        message: str = ""
        score: Optional[float] = None
        threshold: Optional[float] = None
        version: Optional[str] = None
        details: Dict[str, Any] = Field(default_factory=dict)

from .config import (
    DEFAULT_ENTITIES,
    DEFAULT_MAX_TIMEOUT_S,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    MODEL_PATH,
    VERSION,
)
from .engine import PIIRedactionEngine
from .input_adapters import text_to_rows
from .output_formatters import format_entity_summary, format_redacted_text, total_entity_count


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


class PIIRedactionValidator:
    """
    PII redaction validator combining rule-based Presidio recognizers and GLiNER2.

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

    # Uncomment @with_timeout when deploying inside guardrail_services
    # to enable per-request timeout and process-pool isolation.
    # Also change _validate_impl = validate  →  _validate_impl = validate.__wrapped__
    #
    # @with_timeout(
    #     seconds=DEFAULT_MAX_TIMEOUT_S,
    #     pool_manager=_POOL,
    #     config_factory=lambda self: (
    #         self.model_path,
    #         float(self.default_threshold),
    #         tuple(sorted(self.default_entities)),  # must be hashable for pool reuse check
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
            text: Input text to validate. Newline-separated rows are treated as
                individual transcript turns and analysed with surrounding context.
            entities: Entity types to detect. Defaults to DEFAULT_ENTITIES if not
                provided. Pass a subset to restrict detection, or a superset to
                add custom types.
            threshold: Per-call override for the confidence threshold (0–1).
                If not provided, self.default_threshold is used.
            window: Per-call override for the context window size (rows above/below).
                If not provided, self.default_window is used.
            include_redacted_text: If True, the redacted transcript is included in
                details["redacted_text"]. Set to False if only metadata is needed.
            include_entity_breakdown: If True, per-type entity counts are included
                in details["entity_breakdown"].

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

        count = total_entity_count(row_results)
        passed = count == 0
        breakdown = format_entity_summary(row_results)

        reason = (
            f"PII detected: {', '.join(sorted(breakdown.keys()))}."
            if not passed
            else "No PII detected."
        )

        details = {
            "reason": reason,
            "entity_count": count,
            "entities_detected": sorted(breakdown.keys()),
            "model_path": self.model_path,
            "window": eff_window,
            "rows_processed": len(rows),
        }
        if include_redacted_text:
            details["redacted_text"] = format_redacted_text(row_results)
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
    # When @with_timeout is active above, change this to: _validate_impl = validate.__wrapped__
    _validate_impl = validate


def init_pii_redaction_guard(
    model_path: Optional[str] = None,
    default_entities: Optional[List[str]] = None,
    default_threshold: float = DEFAULT_THRESHOLD,
    default_window: int = DEFAULT_WINDOW,
) -> None:
    """Initialize the PII redaction validator singleton at API startup.

    Called once during the FastAPI lifespan. Subsequent calls with the same
    configuration are no-ops. Calling with a different configuration raises
    RuntimeError (restart the service to change the model or defaults).

    Args:
        model_path: Model path override. Falls back to PII_MODEL_PATH env var,
            then to the default HuggingFace model ID in config.py.
        default_entities: Entity types to detect by default. Falls back to
            DEFAULT_ENTITIES from config.py.
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
