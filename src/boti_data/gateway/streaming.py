"""Streaming loads: ``DataGateway.iter_batches`` yields Arrow record batches.

Nothing here materialises the whole result. SQL sources stream through a
server-side cursor (:func:`boti_data.db.arrow_fetch.stream_statement`), Parquet
sources through ``pyarrow.dataset`` batches, and each batch is renamed through
the gateway's field map before it is handed out.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from .load_execution import StreamingSource
from .loaders import _prepare_sql_statement
from .requests import SqlLoadRequest

if TYPE_CHECKING:
    from .core import DataGateway

DEFAULT_BATCH_SIZE = 100_000


def _rename_through_field_map(batch: pa.RecordBatch, field_map: Any) -> pa.RecordBatch:
    if not field_map:
        return batch
    names = [field_map.to_semantic(name) for name in batch.schema.names]
    return batch.rename_columns(names) if names != batch.schema.names else batch


def _sql_batches(source: StreamingSource, batch_size: int) -> Iterator[pa.RecordBatch]:
    from boti_data.db.arrow_fetch import stream_statement

    resource = source.resource
    if resource is None:
        raise RuntimeError("DataGateway.iter_batches() requires a synchronous DSN.")
    statement, params = _prepare_sql_statement(
        SqlLoadRequest.model_validate(source.sql_payload),
        logger=resource.logger,
        debug=resource.debug,
    )
    yield from stream_statement(resource.engine, statement, params=params, batch_size=batch_size)


def _parquet_batches(
    source: StreamingSource, options: dict[str, Any], batch_size: int
) -> Iterator[pa.RecordBatch]:
    import pyarrow.dataset as ds

    from boti_data.parquet import filesystem, schema_filters

    resource = source.resource
    files = resource._resolve_files_to_load()
    if not files:
        return
    dataset = ds.dataset(
        [filesystem.normalized_arrow_load_path(resource, path) for path in files],
        filesystem=filesystem.arrow_filesystem(resource),
        format="parquet",
    )
    expression = schema_filters.raw_filters_to_expression(options.get("raw_filters"))
    yield from dataset.to_batches(
        filter=expression, columns=options.get("columns"), batch_size=batch_size
    )


def iter_batches(gateway: DataGateway, options: dict[str, Any]) -> Iterator[pa.RecordBatch]:
    """Yield the load described by ``options`` as Arrow record batches, lazily."""
    batch_size = int(options.pop("batch_size", DEFAULT_BATCH_SIZE))
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    source = gateway.load_executor.streaming_source(options)
    batches = (
        _parquet_batches(source, options, batch_size)
        if source.backend == "parquet"
        else _sql_batches(source, batch_size)
    )
    for batch in batches:
        yield _rename_through_field_map(batch, source.field_map)
