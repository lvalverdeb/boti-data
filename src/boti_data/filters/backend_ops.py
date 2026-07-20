"""Backend-specific column resolution and operation application (sqlalchemy/dask).

Split out of utils.py purely for line-count headroom. Re-exported from
utils.py so every existing import path keeps working unchanged.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import dask.dataframe as dd
import pandas as pd
from sqlalchemy import Column, String, and_, cast, false, func, or_, true
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import ColumnProperty
from sqlalchemy.sql.sqltypes import Date, Time

from boti_data.filters.value_parsing import (
    DATE_OPERATORS,
    DT_OPERATORS,
    align_in_types,
    as_str,
    normalize_in_filter_values,
)

_log = logging.getLogger(__name__)


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
        raise AttributeError(f"Field '{field_name}' not found in model '{model.__name__}'") from exc

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
        raise AttributeError(f"Field '{field_name}' not found in model '{model.__name__}'")
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


def _istartswith_mask(col: Any, val: Any) -> Any:
    lowered = as_str(col).str.lower()
    return lowered.str.startswith(str(val).lower(), na=False)


def _iendswith_mask(col: Any, val: Any) -> Any:
    lowered = as_str(col).str.lower()
    return lowered.str.endswith(str(val).lower(), na=False)


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
        "regex": lambda col, val: as_str(col).str.contains(val, regex=True, na=False),
        "icontains": lambda col, val: as_str(col).str.contains(
            val, case=False, regex=True, na=False
        ),
        "istartswith": _istartswith_mask,
        "iendswith": _iendswith_mask,
        "iexact": lambda col, val: as_str(col).str.lower() == str(val).lower(),
        "iregex": lambda col, val: as_str(col).str.contains(val, case=False, regex=True, na=False),
        "isnull": lambda col, val: col.isnull() if val else col.notnull(),
        "not_exact": lambda col, val: col != val,
    }


def strip_tz(column: Any) -> Any:
    def _partition(series: pd.Series) -> pd.Series:
        try:
            converted = series.dt.tz_convert("UTC")
            return converted.dt.tz_localize(None)
        except Exception:
            _log.debug("Failed to convert timezone to UTC for series, trying tz_localize(None)")
            try:
                return series.dt.tz_localize(None)
            except Exception:
                _log.debug("Failed to localize timezone for series, returning as-is")
                return series

    return column.map_partitions(_partition, meta=column._meta)


def apply_isin(column: Any, value: Any, negated: bool = False) -> Any:
    aligned_column, aligned_values = align_in_types(column, value)
    mask = aligned_column.isin(aligned_values)
    return ~mask if negated else mask


def _build_in_clauses(
    column: Any,
    normalized_values: list[Any],
    has_null: bool,
    *,
    negated: bool,
) -> list[Any]:
    clauses: list[Any] = []
    if normalized_values:
        clauses.append(~column.in_(normalized_values) if negated else column.in_(normalized_values))
    if has_null:
        clauses.append(column.is_not(None) if negated else column.is_(None))
    return clauses


def apply_in_sqlalchemy(column: Any, value: Any, *, negated: bool) -> Any:
    normalized_values, has_null = normalize_in_filter_values(value)
    if not normalized_values and not has_null:
        return true() if negated else false()
    clauses = _build_in_clauses(column, normalized_values, has_null, negated=negated)
    if len(clauses) == 1:
        return clauses[0]
    combine = and_ if negated else or_
    return combine(*clauses)
