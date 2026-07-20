"""PyArrow compute kernels: string operations (contains/startswith/regex/...).

Split out of arrow_kernels.py purely for god-module headroom. Re-exported
from arrow_kernels.py so every existing import path keeps working unchanged.
"""

from __future__ import annotations

import re

import pyarrow as pa
import pyarrow.compute as pc

from boti_data.filters.arrow_kernels_comparison import ensure_string_array


def _escape_like_pattern(value: str) -> str:
    """Convert a SQL LIKE pattern to a regex pattern."""
    # Escape regex special chars, then convert SQL wildcards
    escaped = re.escape(value)
    # SQL % -> regex .*
    # SQL _ -> regex .
    # But re.escape already escaped them, so we undo that
    escaped = escaped.replace(r"\%", ".*").replace(r"\_", ".")
    return escaped


def contains_kernel(column: pa.ChunkedArray, pattern: str) -> pa.ChunkedArray:
    """column LIKE '%pattern%' (regex=True equivalent)"""
    # Escape special regex characters unless pattern is explicitly a regex
    escaped = _escape_like_pattern(pattern)
    return pc.match_substring_regex(ensure_string_array(column), escaped)


def not_contains_kernel(column: pa.ChunkedArray, pattern: str) -> pa.ChunkedArray:
    """column NOT LIKE '%pattern%'"""
    escaped = _escape_like_pattern(pattern)
    return pc.invert(pc.match_substring_regex(ensure_string_array(column), escaped))


def startswith_kernel(column: pa.ChunkedArray, pattern: str) -> pa.ChunkedArray:
    """column LIKE 'pattern%'"""
    return pc.starts_with(ensure_string_array(column), pattern)


def istartswith_kernel(column: pa.ChunkedArray, pattern: str) -> pa.ChunkedArray:
    """LOWER(column) LIKE LOWER('pattern%')"""
    return pc.starts_with(
        pc.ascii_lower(ensure_string_array(column)),
        pattern.lower(),
    )


def endswith_kernel(column: pa.ChunkedArray, pattern: str) -> pa.ChunkedArray:
    """column LIKE '%pattern'"""
    return pc.ends_with(ensure_string_array(column), pattern)


def iendswith_kernel(column: pa.ChunkedArray, pattern: str) -> pa.ChunkedArray:
    """LOWER(column) LIKE LOWER('%pattern')"""
    return pc.ends_with(
        pc.ascii_lower(ensure_string_array(column)),
        pattern.lower(),
    )


def iexact_kernel(column: pa.ChunkedArray, value: str) -> pa.ChunkedArray:
    """LOWER(column) == LOWER(value)"""
    return pc.equal(
        pc.ascii_lower(ensure_string_array(column)),
        value.lower(),
    )


def icontains_kernel(column: pa.ChunkedArray, pattern: str) -> pa.ChunkedArray:
    """column LIKE '%pattern%' (case-insensitive)"""
    return pc.match_substring_regex(
        ensure_string_array(column),
        _escape_like_pattern(pattern),
        options=pc.MatchSubstringOptions(case_insensitive=True),
    )


def regex_kernel(
    column: pa.ChunkedArray, pattern: str, case_insensitive: bool = False
) -> pa.ChunkedArray:
    """column REGEXP pattern"""
    return pc.match_substring_regex(
        ensure_string_array(column),
        pattern,
        options=pc.MatchSubstringOptions(case_insensitive=case_insensitive),
    )
