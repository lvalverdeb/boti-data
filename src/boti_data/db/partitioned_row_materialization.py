"""Row/frame materialization and schema alignment for SqlPartitionExecutor.

Split out of partitioned_execution.py purely for line-count headroom: these
are pure functions with no dependency on SqlPartitionExecutor's own state
(config/gate_key/use_arrow), so they move here wholesale as free functions.
SqlPartitionExecutor keeps them accessible under their original names via
``staticmethod(...)`` aliases, since tests and sibling modules reference
``SqlPartitionExecutor.align_and_coerce_partition``/
``SqlPartitionExecutor._arrow_align_and_coerce_partition`` directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd
import pyarrow as pa

from boti_data.db.arrow_schema_mapper import (
    arrow_table_to_pandas,
    build_arrow_schema_from_meta_dtypes,
    build_empty_arrow_table,
    coerce_arrow_table,
    rows_to_arrow_table,
)
from boti_data.schema import apply_schema_map

ROW_FETCH_BATCH_SIZE = 100_000


def build_meta_dataframe(meta_dtypes: dict[str, str], *, use_arrow: bool = True) -> pd.DataFrame:
    if use_arrow:
        arrow_schema = build_arrow_schema_from_meta_dtypes(meta_dtypes)
        empty_table = build_empty_arrow_table(arrow_schema)
        return arrow_table_to_pandas(empty_table)
    return pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in meta_dtypes.items()})


def align_and_coerce_partition(
    dataframe: pd.DataFrame,
    meta_dtypes: dict[str, str],
) -> pd.DataFrame:
    expected_columns = list(meta_dtypes)
    actual_columns = list(dataframe.columns)
    if set(actual_columns) != set(expected_columns):
        missing = sorted(set(expected_columns) - set(actual_columns))
        extra = sorted(set(actual_columns) - set(expected_columns))
        raise ValueError(
            f"Partition result columns do not match expected schema. Missing={missing}, extra={extra}."
        )

    ordered = dataframe[expected_columns]
    try:
        aligned = apply_schema_map(
            ordered,
            meta_dtypes,
            require_columns=True,
        )
    except Exception as exc:
        raise ValueError("Failed to align partition output to the expected schema.") from exc
    return aligned


def dataframe_from_result_rows(result: Any, meta_dtypes: dict[str, str]) -> pd.DataFrame:
    columns = list(result.keys())
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    return align_and_coerce_partition(df, meta_dtypes)


def iter_result_batches(
    result: Any,
    *,
    batch_size: int = ROW_FETCH_BATCH_SIZE,
) -> Iterator[list[tuple[Any, ...]]]:
    while True:
        batch = result.fetchmany(batch_size)
        if not batch:
            return
        yield [tuple(row) for row in batch]


def arrow_align_and_coerce_table(
    rows: list[tuple[Any, ...]],
    columns: list[str],
    meta_dtypes: dict[str, str],
) -> pa.Table:
    """Align and coerce partition results into a typed Arrow table."""
    if not rows:
        return build_empty_arrow_table(build_arrow_schema_from_meta_dtypes(meta_dtypes))

    expected_columns = list(meta_dtypes)
    if set(columns) != set(expected_columns):
        missing = sorted(set(expected_columns) - set(columns))
        extra = sorted(set(columns) - set(expected_columns))
        raise ValueError(
            f"Partition result columns do not match expected schema. Missing={missing}, extra={extra}."
        )

    # Build Arrow schema from meta_dtypes and construct table from rows
    arrow_schema = build_arrow_schema_from_meta_dtypes(meta_dtypes)
    # Reorder columns to match expected order
    col_index = {col: idx for idx, col in enumerate(columns)}
    col_order = [col_index[col] for col in expected_columns]
    # Fast path: SQL result already returned columns in schema order
    if col_order == list(range(len(expected_columns))):
        ordered_rows = rows
    else:
        ordered_rows = [tuple(row[i] for i in col_order) for row in rows]

    table = rows_to_arrow_table(ordered_rows, expected_columns, arrow_schema)

    # Coerce to target schema (handles type mismatches)
    try:
        table = coerce_arrow_table(table, arrow_schema)
    except Exception as exc:
        raise ValueError("Failed to align Arrow partition to the expected schema.") from exc
    return table


def arrow_align_and_coerce_partition(
    rows: list[tuple[Any, ...]],
    columns: list[str],
    meta_dtypes: dict[str, str],
) -> pd.DataFrame:
    """Align and coerce partition results using PyArrow for performance."""
    table = arrow_align_and_coerce_table(rows, columns, meta_dtypes)
    return arrow_table_to_pandas(table)


def _materialize_arrow_partition(
    row_batches: Iterator[list[tuple[Any, ...]]],
    columns: list[str],
    meta_dtypes: dict[str, str],
) -> pd.DataFrame:
    tables = [arrow_align_and_coerce_table(batch, columns, meta_dtypes) for batch in row_batches]
    if not tables:
        return build_meta_dataframe(meta_dtypes, use_arrow=True)
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    return arrow_table_to_pandas(table)


def _materialize_pandas_partition(
    row_batches: Iterator[list[tuple[Any, ...]]],
    columns: list[str],
    meta_dtypes: dict[str, str],
) -> pd.DataFrame:
    frames = [
        align_and_coerce_partition(pd.DataFrame(batch, columns=columns), meta_dtypes)
        for batch in row_batches
    ]
    if not frames:
        return build_meta_dataframe(meta_dtypes, use_arrow=False)
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def materialize_partition_from_result(
    result: Any,
    columns: list[str],
    meta_dtypes: dict[str, str],
    *,
    use_arrow: bool,
) -> pd.DataFrame:
    row_batches = iter_result_batches(result)
    if use_arrow:
        return _materialize_arrow_partition(row_batches, columns, meta_dtypes)
    return _materialize_pandas_partition(row_batches, columns, meta_dtypes)
