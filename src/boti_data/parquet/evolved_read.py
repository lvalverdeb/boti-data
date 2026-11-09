"""Lazy reads over a Parquet dataset whose files carry widening schemas.

``ParquetSink`` appends may add columns or promote types (see
:mod:`boti_data.pipelines.sinks_schema`), leaving files of differing schemas
in one dataset. Dask's reader takes the first file's schema as the dataset's,
so a later column goes missing and a promoted type fails the concat. Here each
file is read as a pyarrow fragment against the unified schema: absent columns
come back as nulls and the rest are cast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
import pyarrow as pa
import pyarrow.dataset as ds

from . import filesystem, schema_filters

if TYPE_CHECKING:
    from .resource import ParquetDataResource

__all__ = ["load_files_unified"]


def _fragment_to_pandas(
    fragment: ds.Fragment,
    *,
    schema: pa.Schema,
    expression: ds.Expression | None,
    columns: list[str] | None,
) -> Any:
    table = fragment.to_table(schema=schema, filter=expression, columns=columns)
    return table.to_pandas()


def _open(resource: ParquetDataResource, files: list[str], schema: pa.Schema) -> ds.Dataset:
    base_dir, _ = filesystem.dataset_source(resource)
    return ds.dataset(
        [filesystem.normalized_arrow_load_path(resource, path) for path in files],
        filesystem=filesystem.arrow_filesystem(resource),
        format="parquet",
        schema=schema,
        partitioning=ds.partitioning(flavor="hive"),
        partition_base_dir=filesystem.strip_protocol(base_dir),
    )


def load_files_unified(
    resource: ParquetDataResource,
    files: list[str],
    schema: pa.Schema,
    *,
    filters: list[Any] | None,
    columns: list[str] | None,
) -> dd.DataFrame:
    """One Dask partition per file, every file projected onto *schema*."""
    dataset = _open(resource, files, schema)
    expression = schema_filters.raw_filters_to_expression(filters)
    fragments = list(dataset.get_fragments(expression))
    empty = dataset.schema.empty_table()
    meta = empty.select(columns or dataset.schema.names).to_pandas()
    if not fragments:
        return dd.from_pandas(meta, npartitions=1)
    return dd.from_map(
        _fragment_to_pandas,
        fragments,
        meta=meta,
        schema=dataset.schema,
        expression=expression,
        columns=columns,
    )
