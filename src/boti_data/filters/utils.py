"""Stable import surface for filter parsing/backend helpers.

Implementation lives in value_parsing.py (key/value parsing, coercion) and
backend_ops.py (sqlalchemy/dask column resolution and operation maps),
split out purely for line-count headroom. Re-exported here so every
existing ``from boti_data.filters.utils import ...`` keeps working.
"""

from __future__ import annotations

from boti_data.filters.backend_ops import (
    apply_in_sqlalchemy,
    apply_isin,
    apply_operation_dask,
    apply_operation_sqlalchemy,
    escape_like_pattern,
    get_backend_methods,
    get_dask_column,
    get_sqlalchemy_column,
    operation_map_dask,
    operation_map_sqlalchemy,
    regex_sqlalchemy,
    strip_tz,
)
from boti_data.filters.value_parsing import (
    COMPARISON_OPERATORS,
    DATE_OPERATORS,
    DT_OPERATORS,
    align_in_types,
    as_str,
    normalize_in_filter_values,
    parse_filter_key,
    parse_filter_value,
    pushdown_ops,
    rewrite_date_logic,
    suggest_in_filter_chunking,
    time_to_seconds,
    validate_regex_pattern,
)

__all__ = [
    "COMPARISON_OPERATORS",
    "DATE_OPERATORS",
    "DT_OPERATORS",
    "align_in_types",
    "apply_in_sqlalchemy",
    "apply_isin",
    "apply_operation_dask",
    "apply_operation_sqlalchemy",
    "as_str",
    "escape_like_pattern",
    "get_backend_methods",
    "get_dask_column",
    "get_sqlalchemy_column",
    "normalize_in_filter_values",
    "operation_map_dask",
    "operation_map_sqlalchemy",
    "parse_filter_key",
    "parse_filter_value",
    "pushdown_ops",
    "regex_sqlalchemy",
    "rewrite_date_logic",
    "strip_tz",
    "suggest_in_filter_chunking",
    "time_to_seconds",
    "validate_regex_pattern",
]
