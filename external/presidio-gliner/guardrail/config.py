from typing import List, Optional
import os

# HuggingFace model ID (downloaded on first run) or an absolute local path.
# Override at startup via PII_MODEL_PATH env var or init_pii_redaction_guard(model_path=...).
MODEL_PATH: str = "fastino/gliner2-privacy-filter-PII-multi"

VERSION: str = "v1"

# Default entity types to detect. Callers can pass a subset or superset per request.
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

# Minimum GLiNER2 / Presidio confidence score for a span to be kept.
DEFAULT_THRESHOLD: float = 0.5

# Number of rows above and below the target row to include as context for GLiNER2.
DEFAULT_WINDOW: int = 2

# Hard timeout (seconds) for validate() calls. Reads from env; None = no timeout.
DEFAULT_MAX_TIMEOUT_S: Optional[int] = int(os.getenv("GUARDRAIL_MAX_TIMEOUT_S", 30))
