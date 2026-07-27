"""
input_adapters.py
-----------------
Convert different project input formats into a flat list[str] of rows
so the core engine always receives the same type regardless of caller.

Supported formats:
  - Plain text string  (API default — split on newlines)
  - CSV file bytes     (batch CSV pipeline, e.g. Azure ML)
  - pandas DataFrame   (in-memory DataFrame pipeline)
"""

import io
from typing import List

import pandas as pd


def text_to_rows(text: str) -> List[str]:
    """Split a plain text string into rows on newline boundaries.

    This is the default adapter used when the API receives text: str.
    Each newline-delimited segment becomes one row passed to the engine.

    Args:
        text: Raw input string, e.g. a multi-turn transcript joined by \\n.

    Returns:
        List of row strings. Empty lines are preserved (they may carry positional
        context that GLiNER2's windowing logic needs).
    """
    return text.split("\n")


def csv_bytes_to_rows(data: bytes, text_col: str = "text") -> List[str]:
    """Parse CSV bytes and extract the transcript text column as rows.

    Used by batch pipelines (e.g. Azure ML datastore) that read files as bytes.
    NaN cells are filled with empty strings so the engine never receives None.

    Args:
        data: Raw CSV file content as bytes.
        text_col: Name of the column containing transcript text. Default: "text".

    Returns:
        List of row strings, one per CSV row.

    Raises:
        ValueError: If text_col is not found in the CSV.
    """
    df = pd.read_csv(io.BytesIO(data))
    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found in CSV. Available columns: {list(df.columns)}"
        )
    return df[text_col].fillna("").tolist()


def df_to_rows(df: pd.DataFrame, text_col: str = "text") -> List[str]:
    """Extract the transcript text column from an existing DataFrame as rows.

    Used when the caller already holds a DataFrame in memory (e.g. after a
    merge or filter step) and wants to pass it directly to the engine.

    Args:
        df: Input DataFrame.
        text_col: Name of the column containing transcript text. Default: "text".

    Returns:
        List of row strings, one per DataFrame row.

    Raises:
        ValueError: If text_col is not found in the DataFrame.
    """
    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
        )
    return df[text_col].fillna("").tolist()
