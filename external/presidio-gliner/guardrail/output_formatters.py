"""
output_formatters.py
--------------------
Format RowResult lists into different output shapes for different callers.

The validator.py uses these functions to populate the `details` dict in
GuardrailResult. Callers control which formatters run via the
include_redacted_text and include_entity_breakdown flags on validate().
"""

from collections import Counter
from typing import Dict, List

from .engine import RowResult


def format_redacted_text(row_results: List[RowResult]) -> str:
    """Join redacted rows back into a single string.

    Rows are rejoined with newlines, preserving the same structure as the
    original text passed to text_to_rows().

    Args:
        row_results: List of RowResult from PIIRedactionEngine.analyze_rows().

    Returns:
        Full redacted transcript as a single string.
    """
    return "\n".join(r.redacted for r in row_results)


def format_entity_summary(row_results: List[RowResult]) -> Dict[str, int]:
    """Count detected entities by type across all rows.

    Aggregates spans from every row into a single counter, so the caller
    gets one dict showing how many instances of each entity type were found
    in the full transcript.

    Args:
        row_results: List of RowResult from PIIRedactionEngine.analyze_rows().

    Returns:
        Dict mapping entity type → count, e.g. {"SG_PHONE_NUMBER": 2, "PERSON": 1}.
        Empty dict if no PII was found.
    """
    counter: Counter = Counter()
    for r in row_results:
        for _, _, entity_type in r.spans:
            counter[entity_type] += 1
    return dict(counter)


def total_entity_count(row_results: List[RowResult]) -> int:
    """Return the total number of PII spans detected across all rows.

    Used internally by validator.py to set passed = (count == 0).

    Args:
        row_results: List of RowResult from PIIRedactionEngine.analyze_rows().

    Returns:
        Total span count as an integer.
    """
    return sum(len(r.spans) for r in row_results)
