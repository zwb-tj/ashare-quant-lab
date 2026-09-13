"""Dependency-free markdown table rendering.

``pandas.DataFrame.to_markdown`` requires the optional ``tabulate`` package. A
missing optional dependency must never break a report or a CI run — that is
exactly what happened once, so the project renders its tables itself using only
the standard library.

The renderer is intentionally small: values are formatted for human reading,
numeric columns are right-aligned (``---:``), and pipes/newlines inside cells are
escaped so a symbol or a note can never break the table structure.
"""

from __future__ import annotations

import numbers
from typing import Any

import pandas as pd

__all__ = ["format_cell", "markdown_table"]


def format_cell(value: Any) -> str:
    """Format a single cell for display: no scientific notation, no trailing zeros."""
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        if abs(value) != float("inf") and abs(value - round(value)) < 1e-9 and abs(value) < 1e12:
            return str(round(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def markdown_table(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if df is None or len(df.columns) == 0:
        return ""

    frame = df.reset_index() if index else df

    headers = [str(c) for c in frame.columns]
    numeric_flags = []
    for column in frame.columns:
        values = [v for v in frame[column].tolist() if v is not None and not (isinstance(v, float) and v != v)]
        numeric_flags.append(bool(values) and all(isinstance(v, numbers.Number) and not isinstance(v, bool) for v in values))

    lines = ["| " + " | ".join(h.replace("|", "\\|") for h in headers) + " |"]
    lines.append("| " + " | ".join("---:" if flag else "---" for flag in numeric_flags) + " |")
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(lines)
