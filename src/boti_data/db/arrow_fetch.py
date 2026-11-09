"""Arrow-native partition fetch: cursor results become ``pyarrow.Table`` in one pass.

Two fetchers implement :class:`ArrowFetcher`:

* :class:`NativeArrowFetcher` asks the DBAPI cursor for Arrow directly when it
  can (DuckDB's ``to_arrow_reader``), so no Python row objects exist at all.
* :class:`CursorArrowFetcher` is the fallback for every other driver. It reads
  raw DBAPI rows (bypassing SQLAlchemy's ``Row`` wrapper), transposes a batch
  into columns in C (``zip(*rows)``) and builds each column with one
  ``pyarrow.array`` call, falling back to a vectorised cast and only then to
  the per-value coercion in :mod:`arrow_schema_mapper`.

Both yield ``RecordBatch``es in schema order, so a partition can be streamed
(:meth:`ArrowFetcher.iter_batches`) or materialised (:meth:`fetch_table`)
without a pandas detour. pandas, polars and Dask are views built at the edge.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.compute as pc

from boti_data.db.arrow_schema_mapper import (
    _coerce_value,
    build_arrow_schema_from_meta_dtypes,
)

__all__ = [
    "ArrowFetcher",
    "CursorArrowFetcher",
    "NativeArrowFetcher",
    "ROW_FETCH_BATCH_SIZE",
    "fetch_table",
    "fetcher_for",
    "aselect_batches",
    "iter_record_batches",
    "select_batches",
    "stream_statement",
]

_log = logging.getLogger(__name__)

ROW_FETCH_BATCH_SIZE = 100_000

_TRUE_STRINGS = pa.array(["true", "1", "yes", "y", "t"])


class ArrowFetcher(Protocol):
    """Turn an executed SQL result into Arrow record batches of a known schema."""

    def iter_batches(
        self, result: Any, schema: pa.Schema, *, batch_size: int = ROW_FETCH_BATCH_SIZE
    ) -> Iterator[pa.RecordBatch]: ...

    def fetch_table(
        self, result: Any, schema: pa.Schema, *, batch_size: int = ROW_FETCH_BATCH_SIZE
    ) -> pa.Table: ...


# ---------------------------------------------------------------------------
# Column construction
# ---------------------------------------------------------------------------


def _blank_to_null(values: pa.Array) -> pa.Array:
    trimmed = pc.utf8_trim_whitespace(values)
    return pc.if_else(pc.equal(trimmed, ""), pa.scalar(None, pa.string()), trimmed)


def _strings_to_timestamp(values: pa.Array, target: pa.DataType) -> pa.Array:
    values = _blank_to_null(values)
    if target.tz is None:
        return pc.cast(values, target)
    try:  # strings carrying their own offset parse straight into the zoned type
        return pc.cast(values, target)
    except pa.ArrowInvalid:
        return pc.assume_timezone(pc.cast(values, pa.timestamp(target.unit)), target.tz)


def _cast_timestamp(values: pa.Array, target: pa.DataType) -> pa.Array:
    if pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
        return _strings_to_timestamp(values, target)
    if pa.types.is_timestamp(values.type) and target.tz is not None and values.type.tz is None:
        values = pc.assume_timezone(pc.cast(values, pa.timestamp(target.unit)), target.tz)
    elif pa.types.is_timestamp(values.type) and target.tz is None and values.type.tz is not None:
        values = pc.local_timestamp(values)
    return pc.cast(values, target)


def _cast_boolean(values: pa.Array, target: pa.DataType) -> pa.Array:
    if pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
        lowered = pc.utf8_lower(pc.utf8_trim_whitespace(values))
        truthy = pc.is_in(lowered, value_set=_TRUE_STRINGS)
        return pc.if_else(pc.is_null(values), pa.scalar(None, pa.bool_()), truthy)
    return (
        pc.cast(pc.not_equal(values, 0), target)
        if pa.types.is_integer(values.type)
        else pc.cast(values, target)
    )


def _cast_column(values: pa.Array, target: pa.DataType) -> pa.Array:
    """Vectorised equivalent of the per-value coercion in ``arrow_schema_mapper``."""
    if pa.types.is_timestamp(target):
        return _cast_timestamp(values, target)
    if pa.types.is_boolean(target):
        return _cast_boolean(values, target)
    if pa.types.is_integer(target) or pa.types.is_floating(target):
        return pc.cast(values, target, safe=False)
    return pc.cast(values, target)


def _slow_column(values: tuple[Any, ...], field: pa.Field) -> pa.Array:
    coerced = [_coerce_value(value, field.type) for value in values]
    try:
        return pa.array(coerced, type=field.type)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return pa.array([None if v is None else str(v) for v in values], type=pa.string())


def _direct_or_vectorised(values: tuple[Any, ...], field: pa.Field) -> pa.Array | None:
    try:
        return pa.array(values, type=field.type)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError, OverflowError):
        _log.debug("Direct build failed for %s; trying a vectorised cast", field.name)
    try:
        return _cast_column(pa.array(values), field.type)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, TypeError, ValueError):
        _log.debug("Vectorised cast failed for %s; coercing per value", field.name)
    return None


def column_array(values: tuple[Any, ...], field: pa.Field) -> pa.Array:
    """Build one typed column from Python values: direct, then vectorised, then per value."""
    array = _direct_or_vectorised(values, field)
    return array if array is not None else _slow_column(values, field)


def rows_to_record_batch(
    rows: list[tuple[Any, ...]],
    columns: list[str],
    schema: pa.Schema,
) -> pa.RecordBatch:
    """Transpose ``rows`` (in ``columns`` order) into a batch in ``schema`` order."""
    if not rows:
        return pa.RecordBatch.from_arrays(
            [pa.array([], type=f.type) for f in schema], schema=schema
        )
    by_name = dict(zip(columns, zip(*rows)))
    arrays = [column_array(by_name[field.name], field) for field in schema]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


def _raw_cursor(result: Any) -> Any:
    """The DBAPI cursor behind a SQLAlchemy result, or the result itself.

    Reading the DBAPI cursor skips SQLAlchemy's ``Row`` wrapper, but it is only
    safe while SQLAlchemy has buffered nothing itself: a ``stream_results``
    result pre-fetches rows into its own buffer, so those go through the result.
    """
    cursor = getattr(result, "cursor", None)
    if cursor is None or not hasattr(cursor, "fetchmany"):
        return result
    context = getattr(result, "context", None)
    options = getattr(context, "execution_options", {}) or {}
    if options.get("stream_results") or options.get("yield_per"):
        return result
    return cursor


def _result_columns(result: Any) -> list[str]:
    keys = getattr(result, "keys", None)
    if callable(keys):
        return list(keys())
    return [desc[0] for desc in result.description]


def _check_columns(columns: list[str], schema: pa.Schema) -> None:
    expected = schema.names
    if set(columns) != set(expected):
        missing = sorted(set(expected) - set(columns))
        extra = sorted(set(columns) - set(expected))
        raise ValueError(
            f"Partition result columns do not match expected schema. Missing={missing}, extra={extra}."
        )


def _table_from(fetcher: ArrowFetcher, result: Any, schema: pa.Schema, batch_size: int) -> pa.Table:
    batches = list(fetcher.iter_batches(result, schema, batch_size=batch_size))
    return pa.Table.from_batches(batches, schema=schema)


class CursorArrowFetcher:
    """Columnar one-pass build from raw DBAPI rows; works with every driver."""

    def iter_batches(
        self, result: Any, schema: pa.Schema, *, batch_size: int = ROW_FETCH_BATCH_SIZE
    ) -> Iterator[pa.RecordBatch]:
        columns = _result_columns(result)
        _check_columns(columns, schema)
        cursor = _raw_cursor(result)
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                return
            yield rows_to_record_batch(rows, columns, schema)

    def fetch_table(
        self, result: Any, schema: pa.Schema, *, batch_size: int = ROW_FETCH_BATCH_SIZE
    ) -> pa.Table:
        return _table_from(self, result, schema, batch_size)


class NativeArrowFetcher:
    """Drivers whose cursor yields Arrow itself (DuckDB ``to_arrow_reader``)."""

    @staticmethod
    def supports(result: Any) -> bool:
        return hasattr(_raw_cursor(result), "to_arrow_reader")

    def iter_batches(
        self, result: Any, schema: pa.Schema, *, batch_size: int = ROW_FETCH_BATCH_SIZE
    ) -> Iterator[pa.RecordBatch]:
        reader = _raw_cursor(result).to_arrow_reader(batch_size)
        _check_columns(reader.schema.names, schema)
        for batch in reader:
            table = pa.Table.from_batches([batch]).select(schema.names)
            yield _cast_table(table, schema).combine_chunks().to_batches()[0]

    def fetch_table(
        self, result: Any, schema: pa.Schema, *, batch_size: int = ROW_FETCH_BATCH_SIZE
    ) -> pa.Table:
        return _table_from(self, result, schema, batch_size)


def _cast_table(table: pa.Table, schema: pa.Schema) -> pa.Table:
    if table.schema == schema:
        return table
    arrays = [
        table.column(field.name)
        if table.column(field.name).type == field.type
        else pa.chunked_array(
            [_cast_column(chunk, field.type) for chunk in table.column(field.name).chunks],
            type=field.type,
        )
        for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


_CURSOR_FETCHER = CursorArrowFetcher()
_NATIVE_FETCHER = NativeArrowFetcher()


def fetcher_for(result: Any) -> ArrowFetcher:
    """Pick the native fetcher when the driver offers Arrow, else the cursor fallback."""
    return _NATIVE_FETCHER if NativeArrowFetcher.supports(result) else _CURSOR_FETCHER


def iter_record_batches(
    result: Any,
    meta_dtypes: dict[str, str],
    *,
    batch_size: int = ROW_FETCH_BATCH_SIZE,
) -> Iterator[pa.RecordBatch]:
    """Stream an executed result as typed record batches, never holding the whole result."""
    schema = build_arrow_schema_from_meta_dtypes(meta_dtypes)
    return fetcher_for(result).iter_batches(result, schema, batch_size=batch_size)


def fetch_table(
    result: Any,
    meta_dtypes: dict[str, str],
    *,
    batch_size: int = ROW_FETCH_BATCH_SIZE,
) -> pa.Table:
    """Materialise an executed result as one typed ``pyarrow.Table``."""
    schema = build_arrow_schema_from_meta_dtypes(meta_dtypes)
    return fetcher_for(result).fetch_table(result, schema, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Streaming a statement
# ---------------------------------------------------------------------------


def _schema_for_statement(
    statement: Any, first: list[tuple[Any, ...]], columns: list[str]
) -> pa.Schema:
    """Schema from the statement's column types when it is a Select, else inferred from rows."""
    if hasattr(statement, "selected_columns"):
        from boti_data.db.partitioned_statements import infer_meta_dtypes

        return build_arrow_schema_from_meta_dtypes(infer_meta_dtypes(statement))
    from boti_data.db.arrow_schema_mapper import arrow_table_from_pylist_with_fallback

    return arrow_table_from_pylist_with_fallback(
        [dict(zip(columns, row)) for row in first], columns
    ).schema


def _batches_from_rows(
    cursor: Any, columns: list[str], statement: Any, batch_size: int
) -> Iterator[pa.RecordBatch]:
    first = cursor.fetchmany(batch_size)
    if not first:
        return
    schema = _schema_for_statement(statement, first, columns)
    rows = first
    while rows:
        yield rows_to_record_batch(rows, columns, schema)
        rows = cursor.fetchmany(batch_size)


def stream_statement(
    engine: Any,
    statement: Any,
    *,
    params: dict[str, Any] | None = None,
    batch_size: int = ROW_FETCH_BATCH_SIZE,
) -> Iterator[pa.RecordBatch]:
    """Execute ``statement`` with a server-side cursor and yield typed record batches.

    The connection stays open only while the generator is consumed, and at most
    ``batch_size`` rows are buffered on the client, so a scan of any size costs
    about one batch of memory. Typed from the statement's columns when it is a
    ``Select``; a ``text()`` statement is typed from its first batch.
    """
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True, max_row_buffer=batch_size).execute(
            statement, params or {}
        )
        try:
            if NativeArrowFetcher.supports(result):
                schema = _schema_for_statement(statement, [], list(result.keys()))
                yield from _NATIVE_FETCHER.iter_batches(result, schema, batch_size=batch_size)
                return
            yield from _batches_from_rows(
                _raw_cursor(result), list(result.keys()), statement, batch_size
            )
        finally:
            result.close()


def _select_columns(model: Any, columns: list[str] | None) -> Any:
    from sqlalchemy import select

    if not columns:
        return select(model)
    return select(*(getattr(model, column) for column in columns))


def select_batches(
    engine: Any,
    model: Any,
    *,
    columns: list[str] | None = None,
    batch_size: int = ROW_FETCH_BATCH_SIZE,
) -> Iterator[pa.RecordBatch]:
    """Stream a mapped table (optionally a column subset) as record batches."""
    return stream_statement(engine, _select_columns(model, columns), batch_size=batch_size)


async def aselect_batches(
    engine: Any,
    model: Any,
    *,
    columns: list[str] | None = None,
    batch_size: int = ROW_FETCH_BATCH_SIZE,
) -> AsyncIterator[pa.RecordBatch]:
    """Async twin of :func:`select_batches` over ``AsyncConnection.stream``."""
    statement = _select_columns(model, columns)
    async with engine.connect() as conn:
        result = await conn.stream(statement, execution_options={"max_row_buffer": batch_size})
        names = list(result.keys())
        schema = None
        async for partition in result.partitions(batch_size):
            rows = [tuple(row) for row in partition]
            schema = schema or _schema_for_statement(statement, rows, names)
            yield rows_to_record_batch(rows, names, schema)
