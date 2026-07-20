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


def coerce_temporal_filters(
    resource: ParquetDataResource, filters: dict[str, Any]
) -> dict[str, Any]:
    """Coerce date/datetime filter values to ISO strings for string-typed columns.

    A ``date``/``datetime`` value compared against a string column (e.g. an
    ISO-8601 date stored as text) has no PyArrow comparison kernel and would
    raise ``ArrowNotImplementedError`` on both the pushdown and residual
    paths. Rewriting the value to its ISO string makes the comparison a
    (correct) lexicographic string comparison. Columns with a genuine
    temporal type, and explicitly cast filters, are left untouched.
    """
    if not filters:
        return filters
    string_fields = _string_field_names(resource)
    if not string_fields:
        return filters
    return coerce_mapping(filters, string_fields)


def coerce_mapping(filters: dict[str, Any], string_fields: set[str]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in filters.items():
        if key in {"$and", "$or"} and isinstance(value, (list, tuple)):
            coerced[key] = [
                coerce_mapping(sub, string_fields) if isinstance(sub, dict) else sub
                for sub in value
            ]
        elif key == "$not" and isinstance(value, dict):
            coerced[key] = coerce_mapping(value, string_fields)
        elif str(key).startswith("$"):
            coerced[key] = value
        else:
            field, casting, _op = parse_filter_key(key)
            if casting is None and field in string_fields:
                coerced[key] = stringify_temporal(value)
            else:
                coerced[key] = value
    return coerced


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
