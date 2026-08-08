from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import dask
import dask.dataframe as dd
import pandas as pd
from sqlalchemy.sql import Select

from boti_data.db.partitioned_fetch_gate import (
    GateWaitStats,
    _timed_fetch_gate,
    fetch_gate_stats,
    format_gate_wait_suffix,
    reset_fetch_gate_stats,
)
from boti_data.db.partitioned_row_materialization import (
    align_and_coerce_partition,
    arrow_align_and_coerce_partition,
    arrow_align_and_coerce_table,
    build_meta_dataframe,
    dataframe_from_result_rows,
    iter_result_batches,
    materialize_partition_from_result,
)
from boti_data.db.partitioned_types import SqlPartitionPlan, SqlPartitionSpec
from boti_data.db.sql_config import WorkerSqlConfig
from boti_data.db.sql_engine import (
    _create_worker_async_engine,
    _create_worker_sync_engine,
    _get_worker_engine_identity,
)

# The fetch gate and its contention counters live in partitioned_fetch_gate.py
# (long-file headroom) and are re-exported here so existing import paths — and
# the tests and callers that use them — keep working unchanged.
__all__ = [
    "GateWaitStats",
    "SqlPartitionExecutor",
    "fetch_gate_stats",
    "format_gate_wait_suffix",
    "reset_fetch_gate_stats",
]

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

    def _load_single_partition(
        self,
        plan: SqlPartitionPlan,
        *,
        as_pandas: bool,
        max_concurrent_fetches: int,
        fetch_partition: Callable[..., pd.DataFrame],
        statement: Select | None,
    ) -> pd.DataFrame | dd.DataFrame:
        """Fetch the sole partition inline, skipping the Dask task graph.

        The ``pd.read_sql`` fast read is used only for the non-arrow path (it
        yields a plain pandas frame); the arrow path fetches via
        ``fetch_partition(use_arrow=True)`` so the result matches exactly what
        the delayed multi-partition path would produce.
        """
        if statement is not None and not self.use_arrow:
            engine = _get_cached_worker_sync_engine(self.config)
            with engine.connect() as conn:
                df = pd.read_sql(statement, conn)
            df = SqlPartitionExecutor.align_and_coerce_partition(df, plan.meta_dtypes)
        else:
            partition = plan.partitions[0]
            df = fetch_partition(
                config=self.config,
                gate_key=self.gate_key,
                max_concurrent_fetches=max_concurrent_fetches,
                partition=partition,
                meta_dtypes=plan.meta_dtypes,
                use_arrow=self.use_arrow,
            )
        if as_pandas:
            return df
        return dd.from_pandas(df, npartitions=1)

    def _load_delayed_partitions(
        self,
        plan: SqlPartitionPlan,
        *,
        as_pandas: bool,
        max_concurrent_fetches: int,
        fetch_partition: Callable[..., pd.DataFrame],
        meta_df: pd.DataFrame,
    ) -> pd.DataFrame | dd.DataFrame:
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
        dataframe = dd.from_delayed(delayed_partitions, meta=meta_df, verify_meta=False)
        if as_pandas:
            return dataframe.compute()
        return dataframe

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
            return meta_df if as_pandas else dd.from_pandas(meta_df, npartitions=1)

        if len(plan.partitions) == 1 and fetch_partition is SqlPartitionExecutor.fetch_partition:
            return self._load_single_partition(
                plan,
                as_pandas=as_pandas,
                max_concurrent_fetches=max_concurrent_fetches,
                fetch_partition=fetch_partition,
                statement=statement,
            )

        return self._load_delayed_partitions(
            plan,
            as_pandas=as_pandas,
            max_concurrent_fetches=max_concurrent_fetches,
            fetch_partition=fetch_partition,
            meta_df=meta_df,
        )

    build_meta_dataframe = staticmethod(build_meta_dataframe)
    _dataframe_from_result_rows = staticmethod(dataframe_from_result_rows)

    @staticmethod
    def _partition_exec_args(partition: SqlPartitionSpec) -> tuple[Any, ...]:
        if partition.params is None:
            return (partition.sql,)
        return (partition.sql, partition.params)

    # Not a copy-pasted twin: row-materialization is already shared via
    # _dataframe_from_result_rows()/_materialize_partition_from_result()/
    # _partition_exec_args(); the remaining difference (sync engine.connect()
    # vs a reused-loop async engine + exec_driver_sql await) is an irreducible
    # I/O-boundary difference, deliberately avoiding asyncio.run() overhead.
    @staticmethod
    # spaghetti-ignore[sync-async-duplication]: see above
    def fetch_partition(
        *,
        config: WorkerSqlConfig,
        gate_key: str,
        max_concurrent_fetches: int,
        partition: SqlPartitionSpec,
        meta_dtypes: dict[str, str],
        use_arrow: bool = True,
    ) -> pd.DataFrame:
        with _timed_fetch_gate(gate_key, max_concurrent_fetches):
            engine = _get_cached_worker_sync_engine(config)
            with engine.connect() as conn:
                if not use_arrow:
                    result = conn.exec_driver_sql(partition.sql, partition.params)
                    return SqlPartitionExecutor._dataframe_from_result_rows(result, meta_dtypes)
                result = conn.exec_driver_sql(*SqlPartitionExecutor._partition_exec_args(partition))
                columns = list(result.keys())
                return SqlPartitionExecutor._materialize_partition_from_result(
                    result,
                    columns,
                    meta_dtypes,
                    use_arrow=use_arrow,
                )

    _iter_result_batches = staticmethod(iter_result_batches)
    _materialize_partition_from_result = staticmethod(materialize_partition_from_result)

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
        with _timed_fetch_gate(gate_key, max_concurrent_fetches):

            async def _fetch() -> pd.DataFrame:
                engine = _get_cached_worker_async_engine(config)
                async with engine.connect() as conn:
                    if not use_arrow:
                        result = await conn.exec_driver_sql(partition.sql, partition.params)
                        return SqlPartitionExecutor._dataframe_from_result_rows(result, meta_dtypes)
                    result = await conn.exec_driver_sql(
                        *SqlPartitionExecutor._partition_exec_args(partition)
                    )
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
    def probe_capped(
        sync_conn: Any,
        statement: Select[Any],
        cap: int,
    ) -> pd.DataFrame:
        """Fetch at most ``cap`` rows via a server-side streaming cursor.

        Deliberately does *not* add a SQL ``LIMIT`` to ``statement``: on MySQL a
        ``LIMIT`` on a non-covering filtered scan flips the optimizer to a much
        slower index-driven plan (measured ~3x slower than the same query with no
        ``LIMIT``, even with ``ORDER BY`` or a derived-table wrapper). Streaming
        keeps the fast full-scan plan while still bounding client memory — we read
        at most ``cap`` rows, then close the cursor to abort the rest of the scan.

        Returns an uncoerced frame (column names only); callers align it against
        their ``meta_dtypes`` via :meth:`align_and_coerce_partition`, exactly as
        the batch fetch path does.
        """
        result = sync_conn.execution_options(stream_results=True, max_row_buffer=cap).execute(
            statement
        )
        try:
            rows = [tuple(row) for row in result.fetchmany(cap)]
            columns = list(result.keys())
        finally:
            result.close()
        return pd.DataFrame(rows, columns=columns)

    _arrow_align_and_coerce_table = staticmethod(arrow_align_and_coerce_table)
    _arrow_align_and_coerce_partition = staticmethod(arrow_align_and_coerce_partition)
    align_and_coerce_partition = staticmethod(align_and_coerce_partition)
