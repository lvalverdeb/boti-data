from __future__ import annotations

import datetime
import functools
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

# ---------------------------------------------------------------------------
# ReDoS guard
# ---------------------------------------------------------------------------

# Maximum pattern length accepted for regex/iregex filters.
_MAX_REGEX_PATTERN_LENGTH: int = 500

# Detects structurally dangerous patterns:
#   • Nested quantifiers on a group:  (X+)+  (X*)+ etc.
#   • Quantified alternation group: (a|b)+ or similar with embedded quantifiers
_DANGEROUS_NESTED_QUANTIFIER_RE = re.compile(
    r"\([^()]*[+*][^()]*\)[+*{]"
    r"|"
    r"\([^()]+\)\{"
)


def validate_regex_pattern(pattern: str, *, max_length: int = _MAX_REGEX_PATTERN_LENGTH) -> None:
    """Validate a user-supplied regex pattern for safety before passing to any engine.

    Raises:
        ValueError: If the pattern exceeds *max_length*, is syntactically invalid,
            or contains constructs known to cause catastrophic backtracking.
    """
    if len(pattern) > max_length:
        raise ValueError(
            f"Regex pattern is too long ({len(pattern)} chars); "
            f"maximum allowed is {max_length}."
        )
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {exc}") from exc

    if _DANGEROUS_NESTED_QUANTIFIER_RE.search(pattern):
        raise ValueError(
            "Regex pattern contains nested quantifiers that may cause "
            "catastrophic backtracking and has been rejected for safety. "
            "Simplify the pattern to avoid constructs like (X+)+ or (X*)* ."
        )


def _validated_regex(value: Any) -> str:
    """Validate and return the regex pattern string; raises ValueError on unsafe input."""
    pattern = str(value)
    validate_regex_pattern(pattern)
    return pattern

import dask.dataframe as dd
import pandas as pd
from sqlalchemy import Column, String, and_, cast, false, func, or_, true
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import ColumnProperty
from sqlalchemy.sql.sqltypes import Date, Time

COMPARISON_OPERATORS = [
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
]
DT_OPERATORS = ["date", "time"]
DATE_OPERATORS = ["year", "month", "day", "hour", "minute", "second", "week_day"]


def pushdown_ops() -> set[str]:
    return {"exact", "gt", "gte", "lt", "lte", "in", "range", "not_exact", "not_in"}


def parse_filter_key(key: str) -> tuple[str, str | None, str]:
    parts = key.split("__")
    field_name = parts[0]
    casting = None
    operation = "exact"

    if len(parts) == 3:
        if parts[1] == "not":
            operation = f"not_{parts[2]}"
        else:
            casting = parts[1]
            operation = parts[2]

        if operation == "ne":
            operation = "not_exact"
        if operation == "nin":
            operation = "not_in"

    elif len(parts) == 2:
        second = parts[1]
        if second == "ne":
            operation = "not_exact"
        elif second == "nin":
            operation = "not_in"
        elif second in COMPARISON_OPERATORS:
            operation = second
        elif second in DT_OPERATORS + DATE_OPERATORS:
            casting = second
        else:
            operation = second

    return field_name, casting, operation


def time_to_seconds(value: Any) -> int:
    if isinstance(value, str):
        value = datetime.time.fromisoformat(value)
    return value.hour * 3600 + value.minute * 60 + value.second


def parse_filter_value(casting: str | None, value: Any) -> Any:
    if casting == "date":
        if isinstance(value, str):
            return pd.Timestamp(value)
        if _is_non_string_iterable(value):
            return [pd.Timestamp(item) for item in _coerce_in_values(value)]
    elif casting == "time":
        if _is_non_string_iterable(value):
            return [time_to_seconds(item) for item in _coerce_in_values(value)]
        return time_to_seconds(value)
    return value


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

    if op == "gte":
        return {"gte": date_value}
    if op == "gt":
        return {"gte": date_value + pd.Timedelta(days=1)}
    if op == "lt":
        return {"lt": date_value}
    if op == "lte":
        return {"lt": date_value + pd.Timedelta(days=1)}
    if op == "range":
        start, end = date_value
        return {"range": [start, end + pd.Timedelta(days=1)]}

    raise ValueError(f"Cannot rewrite op {op} for pushdown")


def get_backend_methods(backend: str) -> dict[str, Any]:
    if backend == "sqlalchemy":
        return {
            "get_column": get_sqlalchemy_column,
            "apply_operation": apply_operation_sqlalchemy,
        }
    if backend == "dask":
        return {
            "get_column": get_dask_column,
            "apply_operation": apply_operation_dask,
            "apply_condition": lambda df, condition: df[condition],
        }
    if backend == "arrow":
        return {}
    raise ValueError(f"Unsupported backend: {backend}")


def get_sqlalchemy_column(field_name: str, model: Any, casting: str | None) -> Any:
    mapper = sa_inspect(model)
    try:
        mapped_attr = mapper.attrs[field_name]
    except KeyError as exc:
        raise AttributeError(
            f"Field '{field_name}' not found in model '{model.__name__}'"
        ) from exc

    if not isinstance(mapped_attr, ColumnProperty) or len(mapped_attr.columns) != 1:
        raise AttributeError(
            f"Field '{field_name}' is not a directly mapped column on model '{model.__name__}'"
        )

    if not isinstance(mapped_attr.columns[0], Column):
        raise AttributeError(
            f"Field '{field_name}' is not backed by a concrete SQL column on model '{model.__name__}'"
        )

    column = getattr(model, field_name, None)
    if column is None:
        raise AttributeError(
            f"Field '{field_name}' not found in model '{model.__name__}'"
        )
    if casting == "date":
        column = cast(column, Date)
    elif casting == "time":
        column = cast(column, Time)
    elif casting in DATE_OPERATORS:
        column = func.extract(casting, column)
    return column


def get_dask_column(df: dd.DataFrame, field_name: str, casting: str | None) -> Any:
    if field_name not in df.columns:
        raise AttributeError(f"Field '{field_name}' not found in dataframe.")

    needs_datetime = casting in (DT_OPERATORS + DATE_OPERATORS)
    column = dd.to_datetime(df[field_name], errors="coerce") if needs_datetime else df[field_name]
    if needs_datetime:
        column = strip_tz(column)
    if casting == "date":
        column = column.dt.floor("D")
    elif casting == "time":
        column = column.dt.hour * 3600 + column.dt.minute * 60 + column.dt.second
    elif casting in DATE_OPERATORS:
        attr = "weekday" if casting == "week_day" else casting
        column = getattr(column.dt, attr)
    return column


def apply_operation_sqlalchemy(column: Any, operation: str, value: Any) -> Any:
    operation_map = operation_map_sqlalchemy()
    if operation not in operation_map:
        raise ValueError(f"Unsupported operation: {operation}")
    return operation_map[operation](column, value)


def apply_operation_dask(column: Any, operation: str, value: Any) -> Any:
    operation_map = operation_map_dask()
    if operation not in operation_map:
        raise ValueError(f"Unsupported operation: {operation}")
    return operation_map[operation](column, value)


def escape_like_pattern(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def regex_sqlalchemy(column: Any, value: Any, *, case_insensitive: bool = False) -> Any:
    pattern = str(value)
    validate_regex_pattern(pattern)
    if case_insensitive:
        return func.lower(cast(column, String)).regexp_match(pattern.lower())
    return column.regexp_match(pattern)


@functools.lru_cache(maxsize=1)
def operation_map_sqlalchemy() -> dict[str, Any]:
    return {
        "exact": lambda col, val: col == val,
        "gt": lambda col, val: col > val,
        "gte": lambda col, val: col >= val,
        "lt": lambda col, val: col < val,
        "lte": lambda col, val: col <= val,
        "in": lambda col, val: apply_in_sqlalchemy(col, val, negated=False),
        "range": lambda col, val: col.between(val[0], val[1]),
        "contains": lambda col, val: col.like(f"%{escape_like_pattern(val)}%", escape="\\"),
        "startswith": lambda col, val: col.like(f"{escape_like_pattern(val)}%", escape="\\"),
        "endswith": lambda col, val: col.like(f"%{escape_like_pattern(val)}", escape="\\"),
        "isnull": lambda col, val: col.is_(None) if val else col.isnot(None),
        "not_exact": lambda col, val: col != val,
        "not_contains": lambda col, val: ~col.like(f"%{escape_like_pattern(val)}%", escape="\\"),
        "not_in": lambda col, val: apply_in_sqlalchemy(col, val, negated=True),
        "regex": lambda col, val: regex_sqlalchemy(col, val, case_insensitive=False),
        "icontains": lambda col, val: col.ilike(f"%{escape_like_pattern(val)}%", escape="\\"),
        "istartswith": lambda col, val: col.ilike(f"{escape_like_pattern(val)}%", escape="\\"),
        "iendswith": lambda col, val: col.ilike(f"%{escape_like_pattern(val)}", escape="\\"),
        "iexact": lambda col, val: col.ilike(escape_like_pattern(val), escape="\\"),
        "iregex": lambda col, val: regex_sqlalchemy(col, val, case_insensitive=True),
    }


def operation_map_dask() -> dict[str, Any]:
    return {
        "exact": lambda col, val: col == val,
        "gt": lambda col, val: col > val,
        "gte": lambda col, val: col >= val,
        "lt": lambda col, val: col < val,
        "lte": lambda col, val: col <= val,
        "in": lambda col, val: apply_isin(col, val, negated=False),
        "not_in": lambda col, val: apply_isin(col, val, negated=True),
        "range": lambda col, val: (col >= val[0]) & (col <= val[1]),
        "contains": lambda col, val: as_str(col).str.contains(val, regex=True, na=False),
        "startswith": lambda col, val: as_str(col).str.startswith(val, na=False),
        "endswith": lambda col, val: as_str(col).str.endswith(val, na=False),
        "not_contains": lambda col, val: ~as_str(col).str.contains(val, regex=True, na=False),
        "regex": lambda col, val: as_str(col).str.contains(_validated_regex(val), regex=True, na=False),
        "icontains": lambda col, val: as_str(col).str.contains(val, case=False, regex=True, na=False),
        "istartswith": lambda col, val: as_str(col).str.lower().str.startswith(str(val).lower(), na=False),
        "iendswith": lambda col, val: as_str(col).str.lower().str.endswith(str(val).lower(), na=False),
        "iexact": lambda col, val: as_str(col).str.lower() == str(val).lower(),
        "iregex": lambda col, val: as_str(col).str.contains(_validated_regex(val), case=False, regex=True, na=False),
        "isnull": lambda col, val: col.isnull() if val else col.notnull(),
        "not_exact": lambda col, val: col != val,
    }


def as_str(column: Any) -> Any:
    return column.astype("string").fillna("")


def strip_tz(column: Any) -> Any:
    def _partition(series: pd.Series) -> pd.Series:
        try:
            return series.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            try:
                return series.dt.tz_localize(None)
            except Exception:
                return series

    return column.map_partitions(_partition, meta=column._meta)


def apply_isin(column: Any, value: Any, negated: bool = False) -> Any:
    aligned_column, aligned_values = align_in_types(column, value)
    mask = aligned_column.isin(aligned_values)
    return ~mask if negated else mask


def apply_in_sqlalchemy(column: Any, value: Any, *, negated: bool) -> Any:
    normalized_values, has_null = normalize_in_filter_values(value)
    if negated:
        if not normalized_values and not has_null:
            return true()
        clauses: list[Any] = []
        if normalized_values:
            clauses.append(~column.in_(normalized_values))
        if has_null:
            clauses.append(column.is_not(None))
        if len(clauses) == 1:
            return clauses[0]
        return and_(*clauses)

    if not normalized_values and not has_null:
        return false()
    clauses = []
    if normalized_values:
        clauses.append(column.in_(normalized_values))
    if has_null:
        clauses.append(column.is_(None))
    if len(clauses) == 1:
        return clauses[0]
    return or_(*clauses)


def suggest_in_filter_chunking(
    filters: Mapping[str, Any],
    *,
    chunk_size: int = 900,
    max_concurrency: int = 8,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None

    def _visit(node: Mapping[str, Any]) -> None:
        nonlocal best
        for key, value in node.items():
            if str(key).startswith("$"):
                if isinstance(value, list):
                    for sub in value:
                        if isinstance(sub, Mapping):
                            _visit(sub)
                elif isinstance(value, Mapping):
                    _visit(value)
                continue
            _, _, op = parse_filter_key(str(key))
            if op != "in":
                continue
            normalized_values, _ = normalize_in_filter_values(value)
            value_count = len(normalized_values)
            if value_count <= chunk_size:
                continue
            chunk_count = math.ceil(value_count / chunk_size)
            suggestion = {
                "filter_key": key,
                "value_count": value_count,
                "in_chunk_size": chunk_size,
                "chunk_count": chunk_count,
                "in_chunk_concurrency": max(1, min(max_concurrency, chunk_count)),
            }
            if best is None or suggestion["value_count"] > best["value_count"]:
                best = suggestion

    _visit(filters)
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


def _coerce_in_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (set, frozenset, pd.Index, pd.Series)):
        return list(value)
    to_list = getattr(value, "to_list", None)
    if callable(to_list) and not isinstance(value, (str, bytes)):
        try:
            return list(to_list())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (str, bytes)):
        try:
            result = tolist()
            if isinstance(result, list):
                return result
            if isinstance(result, tuple):
                return list(result)
        except Exception:
            pass
    if _is_non_string_iterable(value):
        return list(value)
    return [value]


def _is_non_string_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping))


def _is_null_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    try:
        return bool(result)
    except Exception:
        return False


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
            pass
    if kind in ("f",):
        try:
            return column.astype("float64"), [float(item) for item in values]
        except Exception:
            pass
    return as_str(column), [str(item) for item in values]
