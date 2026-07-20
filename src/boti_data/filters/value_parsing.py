"""Filter-key parsing and value normalization/coercion.

Split out of utils.py purely for line-count headroom: these are the pure
key-parsing/value-coercion helpers, with no dependency on any backend
(sqlalchemy/dask) machinery. Re-exported from utils.py so every existing
import path keeps working unchanged.
"""

from __future__ import annotations

import datetime
import logging
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)

COMPARISON_OPERATORS = (
    "gte",
    "lte",
    "gt",
    "lt",
    "exact",
    "in",
    "range",
    "contains",
    "startswith",
    "endswith",
    "isnull",
    "not_exact",
    "not_contains",
    "not_in",
    "regex",
    "icontains",
    "istartswith",
    "iendswith",
    "iexact",
    "iregex",
)
DT_OPERATORS = ("date", "time")
DATE_OPERATORS = ("year", "month", "day", "hour", "minute", "second", "week_day")


_RE_NESTED_QUANTIFIER = re.compile(r"\([^)]*[+*?}{]\)[+*?{]")


def validate_regex_pattern(pattern: str) -> None:
    if len(pattern) > 500:
        raise ValueError(f"Regex pattern is too long ({len(pattern)} chars, max 500).")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {exc}") from exc
    if _RE_NESTED_QUANTIFIER.search(pattern):
        raise ValueError(
            "Regex pattern contains nested quantifiers which may cause "
            "catastrophic backtracking (ReDoS)."
        )


def pushdown_ops() -> set[str]:
    return {"exact", "gt", "gte", "lt", "lte", "in", "range", "not_exact", "not_in"}


def _normalize_op(op: str) -> str:
    if op == "ne":
        return "not_exact"
    if op == "nin":
        return "not_in"
    return op


def _parse_3part_key(parts: list[str]) -> tuple[str | None, str]:
    if parts[1] == "not":
        return None, _normalize_op(f"not_{parts[2]}")
    return parts[1], _normalize_op(parts[2])


def _parse_2part_key(second: str) -> tuple[str | None, str]:
    if second in COMPARISON_OPERATORS:
        return None, second
    if second in DT_OPERATORS + DATE_OPERATORS:
        return second, "exact"
    return None, _normalize_op(second)


def parse_filter_key(key: str) -> tuple[str, str | None, str]:
    parts = key.split("__")
    field_name = parts[0]
    if len(parts) == 3:
        casting, operation = _parse_3part_key(parts)
        return field_name, casting, operation
    if len(parts) == 2:
        casting, operation = _parse_2part_key(parts[1])
        return field_name, casting, operation
    return field_name, None, "exact"


def time_to_seconds(value: Any) -> int:
    if isinstance(value, str):
        value = datetime.time.fromisoformat(value)
    return value.hour * 3600 + value.minute * 60 + value.second


def _parse_date_filter_value(value: Any) -> Any:
    if isinstance(value, str):
        return pd.Timestamp(value)
    if _is_non_string_iterable(value):
        return [pd.Timestamp(item) for item in _coerce_in_values(value)]
    return value


def _parse_time_filter_value(value: Any) -> Any:
    if _is_non_string_iterable(value):
        return [time_to_seconds(item) for item in _coerce_in_values(value)]
    return time_to_seconds(value)


def parse_filter_value(casting: str | None, value: Any) -> Any:
    if casting == "date":
        return _parse_date_filter_value(value)
    if casting == "time":
        return _parse_time_filter_value(value)
    return value


def _rewrite_range_date_logic(date_value: Any) -> dict[str, Any]:
    start, end = date_value
    return {"range": [start, end + pd.Timedelta(days=1)]}


_DATE_LOGIC_REWRITERS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "gte": lambda date_value: {"gte": date_value},
    "gt": lambda date_value: {"gte": date_value + pd.Timedelta(days=1)},
    "lt": lambda date_value: {"lt": date_value},
    "lte": lambda date_value: {"lt": date_value + pd.Timedelta(days=1)},
    "range": _rewrite_range_date_logic,
}


def rewrite_date_logic(op: str, value: Any) -> dict[str, Any]:
    def _to_utc_timestamp(item: Any) -> pd.Timestamp:
        timestamp = pd.Timestamp(item)
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.normalize()

    if isinstance(value, (list, tuple)):
        date_value = [_to_utc_timestamp(item) for item in value]
    else:
        date_value = _to_utc_timestamp(value)

    rewriter = _DATE_LOGIC_REWRITERS.get(op)
    if rewriter is None:
        raise ValueError(f"Cannot rewrite op {op} for pushdown")
    return rewriter(date_value)


def _suggest_walk_filters(
    node: Mapping[str, Any],
    *,
    chunk_size: int = 900,
    max_concurrency: int = 8,
) -> Iterator[dict[str, Any]]:
    for key, value in node.items():
        if str(key).startswith("$"):
            items = value if isinstance(value, list) else [value]
            for sub in items:
                if isinstance(sub, Mapping):
                    yield from _suggest_walk_filters(
                        sub, chunk_size=chunk_size, max_concurrency=max_concurrency
                    )
            continue
        _, _, op = parse_filter_key(str(key))
        if op != "in":
            continue
        normalized_values, _ = normalize_in_filter_values(value)
        if len(normalized_values) <= chunk_size:
            continue
        chunk_count = math.ceil(len(normalized_values) / chunk_size)
        yield {
            "filter_key": key,
            "value_count": len(normalized_values),
            "in_chunk_size": chunk_size,
            "chunk_count": chunk_count,
            "in_chunk_concurrency": max(1, min(max_concurrency, chunk_count)),
        }


def suggest_in_filter_chunking(
    filters: Mapping[str, Any],
    *,
    chunk_size: int = 900,
    max_concurrency: int = 8,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for suggestion in _suggest_walk_filters(
        filters, chunk_size=chunk_size, max_concurrency=max_concurrency
    ):
        if best is None or suggestion["value_count"] > best["value_count"]:
            best = suggestion
    return best


def normalize_in_filter_values(value: Any) -> tuple[list[Any], bool]:
    values = _coerce_in_values(value)
    normalized: list[Any] = []
    seen_hashable: set[Any] = set()
    seen_unhashable: list[Any] = []
    has_null = False

    for item in values:
        if _is_null_scalar(item):
            has_null = True
            continue
        try:
            if item in seen_hashable:
                continue
            seen_hashable.add(item)
        except TypeError:
            if any(item == existing for existing in seen_unhashable):
                continue
            seen_unhashable.append(item)
        normalized.append(item)

    return normalized, has_null


def _try_to_list(value: Any) -> list[Any] | None:
    for attr in ("to_list", "tolist"):
        fn = getattr(value, attr, None)
        if callable(fn) and not isinstance(value, (str, bytes)):
            try:
                result = fn()
                if isinstance(result, (list, tuple)):
                    return list(result)
            except Exception:
                _log.debug("Failed to call %s on value of type %s", attr, type(value).__name__)
    return None


def _coerce_in_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set, frozenset, pd.Index, pd.Series)):
        return list(value)
    result = _try_to_list(value)
    if result is not None:
        return result
    return list(value) if _is_non_string_iterable(value) else [value]


def _is_non_string_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping))


_NULL_CHECK_FAILED = object()


def _try_pd_isna(value: Any) -> Any:
    try:
        return pd.isna(value)
    except Exception:
        _log.debug("pd.isna failed for value of type %s", type(value).__name__)
        return _NULL_CHECK_FAILED


def _coerce_isna_bool(result: Any, value: Any) -> bool:
    try:
        return bool(result)
    except Exception:
        _log.debug(
            "bool() conversion failed for pd.isna result on value of type %s", type(value).__name__
        )
        return False


def _is_null_scalar(value: Any) -> bool:
    if value is None:
        return True
    result = _try_pd_isna(value)
    if result is _NULL_CHECK_FAILED:
        return False
    return _coerce_isna_bool(result, value)


def align_in_types(column: Any, value: Any) -> tuple[Any, list[Any]]:
    if isinstance(value, (set, tuple)):
        values = list(value)
    elif isinstance(value, list):
        values = value
    else:
        values = [value]

    kind = getattr(getattr(column, "dtype", None), "kind", None)
    if kind in ("i", "u"):
        try:
            return column.astype("Int64"), [int(item) for item in values]
        except Exception:
            _log.debug("Failed to cast column to Int64, falling back to string")
    if kind in ("f",):
        try:
            return column.astype("float64"), [float(item) for item in values]
        except Exception:
            _log.debug("Failed to cast column to float64, falling back to string")
    return as_str(column), [str(item) for item in values]


def as_str(column: Any) -> Any:
    return column.astype("string").fillna("")
