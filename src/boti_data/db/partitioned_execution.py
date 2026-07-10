from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterator
from typing import Any

import dask
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from sqlalchemy.sql import Select

from boti_data.db.arrow_schema_mapper import (
    arrow_table_to_pandas,
    build_arrow_schema_from_meta_dtypes,
    build_empty_arrow_table,
    coerce_arrow_table,
    rows_to_arrow_table,
)
from boti_data.db.partitioned_types import SqlPartitionPlan, SqlPartitionSpec
from boti_data.db.sql_config import WorkerSqlConfig
from boti_data.db.sql_engine import (
    _create_worker_async_engine,
    _create_worker_sync_engine,
    _get_worker_engine_identity,
)
from boti_data.schema import apply_schema_map

_FETCH_GATES: dict[tuple[str, int], threading.BoundedSemaphore] = {}
_FETCH_GATES_LOCK = threading.Lock()
_ROW_FETCH_BATCH_SIZE = 100_000

# Thread-local caches: avoid recreating engine objects and event loops on every partition fetch.
# The engine itself is cheap state (dialect, URL, event listeners); caching it per worker
# thread eliminates repeated URL parsing, driver validation, and read-only listener setup.
# The async event loop is expensive to create; reusing it across run_until_complete() calls
# (NullPool means no persistent connection pool state that binds to a specific loop).
_WORKER_SYNC_ENGINE_LOCAL = threading.local()
_WORKER_ASYNC_LOCAL = threading.local()


def _get_cached_worker_sync_engine(config: WorkerSqlConfig) -> Any:
    """Return a thread-local cached sync engine, creating it on first use."""
    cache = getattr(_WORKER_SYNC_ENGINE_LOCAL, "engines", None)
    if cache is None:
        _WORKER_SYNC_ENGINE_LOCAL.engines = {}
        cache = _WORKER_SYNC_ENGINE_LOCAL.engines
    key = _get_worker_engine_identity(config)
    engine = cache.get(key)
    if engine is None:
        engine = _create_worker_sync_engine(config)
        cache[key] = engine
    return engine


def _get_or_create_worker_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent per-thread event loop, creating it on first use or after closure."""
    loop: asyncio.AbstractEventLoop | None = getattr(_WORKER_ASYNC_LOCAL, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _WORKER_ASYNC_LOCAL.loop = loop
    return loop


def _get_cached_worker_async_engine(config: WorkerSqlConfig) -> Any:
    """Return a thread-local cached async engine tied to the thread's persistent event loop."""
    engines = getattr(_WORKER_ASYNC_LOCAL, "engines", None)
    if engines is None:
        _WORKER_ASYNC_LOCAL.engines = {}
        engines = _WORKER_ASYNC_LOCAL.engines
    key = _get_worker_engine_identity(config)
    engine = engines.get(key)
    if engine is None:
        engine = _create_worker_async_engine(config)
        engines[key] = engine
    return engine


def _get_fetch_gate(gate_key: str, max_concurrent_fetches: int) -> threading.BoundedSemaphore:
    cache_key = (gate_key, max_concurrent_fetches)
    with _FETCH_GATES_LOCK:
        gate = _FETCH_GATES.get(cache_key)
        if gate is None:
            gate = threading.BoundedSemaphore(max_concurrent_fetches)
            _FETCH_GATES[cache_key] = gate
        return gate


class SqlPartitionExecutor:
    """Execute planned partitions and assemble lazy Dask outputs."""

    def __init__(
        self,
        config: WorkerSqlConfig,
        gate_key: str,
        *,
        use_arrow: bool = True,
    ) -> None:
        self.config = config
        self.gate_key = gate_key
        self.use_arrow = use_arrow

    def load_plan(
        self,
        plan: SqlPartitionPlan,
        *,
        as_pandas: bool,
        max_concurrent_fetches: int,
        fetch_partition: Callable[..., pd.DataFrame],
        statement: Select | None = None,
    ) -> pd.DataFrame | dd.DataFrame:
        meta_df = self.build_meta_dataframe(plan.meta_dtypes, use_arrow=self.use_arrow)

        if not plan.partitions:
            if as_pandas:
                return meta_df
            return dd.from_pandas(meta_df, npartitions=1)

        if (
            len(plan.partitions) == 1
            and not self.use_arrow
            and fetch_partition is SqlPartitionExecutor.fetch_partition
        ):
            if statement is not None:
                engine = _get_cached_worker_sync_engine(self.config)
                with engine.connect() as conn:
                    df = pd.read_sql(statement, conn)
                df = SqlPartitionExecutor.align_and_coerce_partition(
                    df, plan.meta_dtypes,
                )
            else:
                partition = plan.partitions[0]
                df = fetch_partition(
                    config=self.config,
                    gate_key=self.gate_key,
                    max_concurrent_fetches=max_concurrent_fetches,
                    partition=partition,
                    meta_dtypes=plan.meta_dtypes,
                    use_arrow=False,
                )
            if as_pandas:
                return df
            return dd.from_pandas(df, npartitions=1)

        delayed_partitions = [
            dask.delayed(fetch_partition)(
                config=self.config,
                gate_key=self.gate_key,
                max_concurrent_fetches=max_concurrent_fetches,
                partition=partition,
                meta_dtypes=plan.meta_dtypes,
                use_arrow=self.use_arrow,
            )
            for partition in plan.partitions
        ]

        dataframe = dd.from_delayed(
            delayed_partitions,
            meta=meta_df,
            verify_meta=False,
        )
        if as_pandas:
            return dataframe.compute()
        return dataframe

    @staticmethod
    def build_meta_dataframe(
        meta_dtypes: dict[str, str], *, use_arrow: bool = True
    ) -> pd.DataFrame:
        if use_arrow:
            arrow_schema = build_arrow_schema_from_meta_dtypes(meta_dtypes)
            empty_table = build_empty_arrow_table(arrow_schema)
            return arrow_table_to_pandas(empty_table)
        return pd.DataFrame(
            {column: pd.Series(dtype=dtype) for column, dtype in meta_dtypes.items()}
        )

    @staticmethod
    def _dataframe_from_result_rows(result: Any, meta_dtypes: dict[str, str]) -> pd.DataFrame:
        columns = list(result.keys())
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=columns)
        return SqlPartitionExecutor.align_and_coerce_partition(df, meta_dtypes)

    @staticmethod
    def fetch_partition(
        *,
        config: WorkerSqlConfig,
        gate_key: str,
        max_concurrent_fetches: int,
        partition: SqlPartitionSpec,
        meta_dtypes: dict[str, str],
        use_arrow: bool = True,
    ) -> pd.DataFrame:
        gate = _get_fetch_gate(gate_key, max_concurrent_fetches)
        with gate:
            engine = _get_cached_worker_sync_engine(config)
            with engine.connect() as conn:
                if not use_arrow:
                    result = conn.exec_driver_sql(partition.sql, partition.params)
                    return SqlPartitionExecutor._dataframe_from_result_rows(result, meta_dtypes)
                if partition.params is None:
                    result = conn.exec_driver_sql(partition.sql)
                else:
                    result = conn.exec_driver_sql(partition.sql, partition.params)
                columns = list(result.keys())
                return SqlPartitionExecutor._materialize_partition_from_result(
                    result,
                    columns,
                    meta_dtypes,
                    use_arrow=use_arrow,
                )

    @staticmethod
    def _iter_result_batches(
        result: Any,
        *,
        batch_size: int = _ROW_FETCH_BATCH_SIZE,
    ) -> Iterator[list[tuple[Any, ...]]]:
        while True:
            batch = result.fetchmany(batch_size)
            if not batch:
                return
            yield [tuple(row) for row in batch]

    @staticmethod
    def _materialize_partition_from_result(
        result: Any,
        columns: list[str],
        meta_dtypes: dict[str, str],
        *,
        use_arrow: bool,
    ) -> pd.DataFrame:
        row_batches = SqlPartitionExecutor._iter_result_batches(result)

        if use_arrow:
            tables = [
                SqlPartitionExecutor._arrow_align_and_coerce_table(
                    batch,
                    columns,
                    meta_dtypes,
                )
                for batch in row_batches
            ]
            if not tables:
                return SqlPartitionExecutor.build_meta_dataframe(meta_dtypes, use_arrow=True)
            table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
            return arrow_table_to_pandas(table)

        frames = [
            SqlPartitionExecutor.align_and_coerce_partition(
                pd.DataFrame(batch, columns=columns),
                meta_dtypes,
            )
            for batch in row_batches
        ]
        if not frames:
            return SqlPartitionExecutor.build_meta_dataframe(meta_dtypes, use_arrow=False)
        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    @staticmethod
    def fetch_partition_async(
        *,
        config: WorkerSqlConfig,
        gate_key: str,
        max_concurrent_fetches: int,
        partition: SqlPartitionSpec,
        meta_dtypes: dict[str, str],
        use_arrow: bool = True,
    ) -> pd.DataFrame:
        """Fetch a single partition using an async DB driver.

        Uses a persistent per-thread event loop (``loop.run_until_complete()``) and a
        cached async engine instead of ``asyncio.run()`` to avoid the overhead of
        creating and tearing down a new event loop on every partition fetch.
        NullPool means the engine carries no loop-bound connection pool state, so
        both the loop and the engine are safe to reuse across calls in the same thread.
        """
        gate = _get_fetch_gate(gate_key, max_concurrent_fetches)
        with gate:

            async def _fetch() -> pd.DataFrame:
                engine = _get_cached_worker_async_engine(config)
                async with engine.connect() as conn:
                    if not use_arrow:
                        result = await conn.exec_driver_sql(partition.sql, partition.params)
                        return SqlPartitionExecutor._dataframe_from_result_rows(result, meta_dtypes)
                    if partition.params is None:
                        result = await conn.exec_driver_sql(partition.sql)
                    else:
                        result = await conn.exec_driver_sql(partition.sql, partition.params)
                    columns = list(result.keys())
                    return SqlPartitionExecutor._materialize_partition_from_result(
                        result,
                        columns,
                        meta_dtypes,
                        use_arrow=use_arrow,
                    )

            loop = _get_or_create_worker_loop()
            return loop.run_until_complete(_fetch())

    @staticmethod
    def _arrow_align_and_coerce_table(
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

    @staticmethod
    def _arrow_align_and_coerce_partition(
        rows: list[tuple[Any, ...]],
        columns: list[str],
        meta_dtypes: dict[str, str],
    ) -> pd.DataFrame:
        """Align and coerce partition results using PyArrow for performance."""
        table = SqlPartitionExecutor._arrow_align_and_coerce_table(
            rows,
            columns,
            meta_dtypes,
        )
        return arrow_table_to_pandas(table)

    @staticmethod
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
