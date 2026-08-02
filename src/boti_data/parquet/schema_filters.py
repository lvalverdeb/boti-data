"""Schema caching, temporal-filter coercion, and raw filter-clause building for ParquetDataResource."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from boti_data.filters.utils import parse_filter_key

from . import filesystem

if TYPE_CHECKING:
    from .resource import ParquetDataResource

_RAW_FILTER_CLAUSE_DISPATCH: dict[str, Callable[[Any, Any], ds.Expression]] = {
    "=": lambda col, v: col == v,
    "!=": lambda col, v: col != v,
    ">": lambda col, v: col > v,
    ">=": lambda col, v: col >= v,
    "<": lambda col, v: col < v,
    "<=": lambda col, v: col <= v,
    "in": lambda col, v: col.isin(v),
    "not in": lambda col, v: ~col.isin(v),
}


def _load_dataset_schema(resource: ParquetDataResource, files: list[str]) -> pa.Schema | None:
    try:
        dataset = ds.dataset(
            [filesystem.normalized_arrow_load_path(resource, path) for path in files],
            filesystem=filesystem.arrow_filesystem(resource),
            format="parquet",
        )
        return dataset.schema
    except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException):
        return None


def dataset_schema(resource: ParquetDataResource) -> pa.Schema | None:
    """Return the Arrow schema of the resolved dataset, or ``None`` if unavailable.

    Cached per-instance: the storage schema is stable for a fixed config, and
    this spares a metadata round-trip on every filtered load.
    """
    cached = getattr(resource, "_schema_cache", None)
    if cached is not None:
        return cached
    files = resource._resolve_files_to_load()
    if not files:
        return None
    schema = _load_dataset_schema(resource, files)
    if schema is not None:
        resource._schema_cache = schema
    return schema


def _string_field_names(resource: ParquetDataResource) -> set[str] | None:
    """Returns the dataset's string-typed column names, or None if unavailable/empty."""
    schema = dataset_schema(resource)
    if schema is None:
        return None
    string_fields = {
        field.name
        for field in schema
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    }
    return string_fields or None


def _temporal_field_tzs(resource: ParquetDataResource) -> dict[str, str | None]:
    """Returns {field_name: tz} for schema fields with a genuine PyArrow
    timestamp type (``tz`` is ``None`` for a naive/no-tz column)."""
    schema = dataset_schema(resource)
    if schema is None:
        return {}
    return {field.name: field.type.tz for field in schema if pa.types.is_timestamp(field.type)}


def coerce_temporal_filters(
    resource: ParquetDataResource, filters: dict[str, Any]
) -> dict[str, Any]:
    """Coerce filter values so their type matches the target column's PyArrow type.

    Two mismatches share the same underlying failure mode — PyArrow's
    filter-pushdown has no comparison kernel between mismatched temporal
    representations, raising ``ArrowNotImplementedError``/``ArrowInvalid``
    on both the pushdown and residual paths:

    - A ``date``/``datetime`` value compared against a **string** column
      (e.g. an ISO-8601 date stored as text) is rewritten to its ISO string,
      making the comparison a correct lexicographic string comparison.
    - A bare ``datetime.date`` value compared against a genuine **timestamp**
      column (tz-aware or not) is promoted to a ``pd.Timestamp`` carrying
      the column's own tz, since callers commonly build period filters via
      ``pd.to_datetime(x).date()`` regardless of the target column's actual
      precision (see ``prepare_period_filters()`` in ``gateway/normalization.py``).

    Explicitly cast filters (``field__date__gte=...``) and values that
    already carry full datetime precision are left untouched either way.
    """
    if not filters:
        return filters
    string_fields = _string_field_names(resource) or set()
    temporal_field_tzs = _temporal_field_tzs(resource)
    if not string_fields and not temporal_field_tzs:
        return filters
    return coerce_mapping(filters, string_fields, temporal_field_tzs)


def coerce_mapping(
    filters: dict[str, Any],
    string_fields: set[str],
    temporal_field_tzs: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    temporal_field_tzs = temporal_field_tzs or {}
    coerced: dict[str, Any] = {}
    for key, value in filters.items():
        if key in {"$and", "$or"} and isinstance(value, (list, tuple)):
            coerced[key] = [
                coerce_mapping(sub, string_fields, temporal_field_tzs)
                if isinstance(sub, dict)
                else sub
                for sub in value
            ]
        elif key == "$not" and isinstance(value, dict):
            coerced[key] = coerce_mapping(value, string_fields, temporal_field_tzs)
        elif str(key).startswith("$"):
            coerced[key] = value
        else:
            coerced[key] = _coerce_single_filter_value(
                key, value, string_fields, temporal_field_tzs
            )
    return coerced


def _coerce_single_filter_value(
    key: str,
    value: Any,
    string_fields: set[str],
    temporal_field_tzs: dict[str, str | None],
) -> Any:
    field, casting, _op = parse_filter_key(key)
    if casting is not None:
        return value
    if field in string_fields:
        return stringify_temporal(value)
    if field in temporal_field_tzs:
        return promote_bare_date_to_timestamp(value, temporal_field_tzs[field])
    return value


def stringify_temporal(value: Any) -> Any:
    def convert(item: Any) -> Any:
        # pandas.Timestamp subclasses datetime, so it is handled here too.
        if isinstance(item, dt.datetime):
            return item.date().isoformat() if item.time() == dt.time(0, 0) else item.isoformat()
        if isinstance(item, dt.date):
            return item.isoformat()
        return item

    if isinstance(value, (list, tuple)):
        return type(value)(convert(item) for item in value)
    return convert(value)


def promote_bare_date_to_timestamp(value: Any, tz: str | None) -> Any:
    """Promote a bare ``datetime.date`` to a ``pd.Timestamp`` in ``tz``.

    Values already carrying datetime precision (``pd.Timestamp``/``datetime.datetime``,
    which subclasses ``date``) are left untouched — only a truly bare
    ``date`` needs promotion to satisfy PyArrow's timestamp comparison kernel.
    """

    def convert(item: Any) -> Any:
        if isinstance(item, dt.datetime):
            return item
        if isinstance(item, dt.date):
            ts = pd.Timestamp(item)
            return ts.tz_localize(tz) if tz else ts
        return item

    if isinstance(value, (list, tuple)):
        return type(value)(convert(item) for item in value)
    return convert(value)


def raw_filters_to_expression(filters: list[Any] | None) -> ds.Expression | None:
    if not filters:
        return None

    expression: ds.Expression | None = None
    for field, op, value in filters:
        clause = raw_filter_clause(field, op, value)
        expression = clause if expression is None else expression & clause
    return expression


def raw_filter_clause(field: str, op: str, value: Any) -> ds.Expression:
    column = ds.field(field)
    handler = _RAW_FILTER_CLAUSE_DISPATCH.get(op)
    if handler is None:
        raise ValueError(f"Unsupported parquet pushdown operator for Arrow load: {op!r}")
    return handler(column, value)


def empty_ddf() -> dd.DataFrame:
    return dd.from_pandas(pd.DataFrame(), npartitions=1)
