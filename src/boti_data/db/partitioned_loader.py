from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

import dask.dataframe as dd
import pandas as pd
from boti.core.lifecycle import LifecycleCore
from boti.core.lifecycle_pickle import PicklableLifecycleCoreMixin
from boti.core.logger import Logger
from sqlalchemy.engine import url as sqlalchemy_url

from boti_data.db.partitioned_execution import SqlPartitionExecutor
from boti_data.db.partitioned_planner import SqlPartitionPlanner
from boti_data.db.partitioned_types import (
    SqlPartitionedLoadRequest,
    SqlPartitionPlan,
)
from boti_data.db.sql_config import SqlDatabaseConfig, WorkerSqlConfig
from boti_data.db.sql_engine import _get_worker_engine_identity
from boti_data.db.sql_resource import SqlDatabaseResource


class SqlPartitionedLoader(PicklableLifecycleCoreMixin, LifecycleCore):
    """Partition-aware SQL loader that returns a lazy Dask DataFrame."""

    def __init__(
        self,
        config: SqlDatabaseConfig,
        *,
        resource: SqlDatabaseResource | None = None,
        use_arrow: bool = True,
    ) -> None:
        self.config = config
        self._resource = resource or SqlDatabaseResource(config)
        self._owns_resource = resource is None
        self._worker_config = WorkerSqlConfig.from_database_config(config)
        self._gate_key = _get_worker_engine_identity(self._worker_config)
        self._planner = SqlPartitionPlanner(self.resource)
        self._executor = SqlPartitionExecutor(
            self._worker_config,
            self._gate_key,
            use_arrow=use_arrow,
        )
        self._use_arrow = use_arrow
        self._validate_distributed_config()
        super().__init__()

    @property
    def resource(self) -> SqlDatabaseResource:
        return self._resource

    @property
    def logger(self) -> Any:
        return self.resource.logger

    def _validate_distributed_config(self) -> None:
        """Reject connection URLs that cannot be reopened safely by worker-local engines."""
        parsed = sqlalchemy_url.make_url(self.config.connection_url.get_secret_value())
        if parsed.get_backend_name() != "sqlite":
            return

        database = parsed.database or ""
        if database == "" or ":memory:" in database:
            raise ValueError(
                "SqlPartitionedLoader requires a database reachable by worker-local connections. "
                "SQLite in-memory DSNs are not supported; use a file-backed DSN or request as_pandas=True."
            )

    def _cleanup(self) -> None:
        if self._owns_resource:
            self.resource.close()

    async def _acleanup(self) -> None:
        # Previously missing entirely: SqlPartitionedLoader had no aclose()/
        # __aenter__/__aexit__ at all, even though the wrapped SqlDatabaseResource
        # (a ManagedResource) has always supported async cleanup.
        if self._owns_resource:
            await self.resource.aclose()

    @staticmethod
    def _coerce_request(
        request: SqlPartitionedLoadRequest | None,
        options: dict[str, Any],
    ) -> SqlPartitionedLoadRequest:
        if request is not None:
            if options:
                raise TypeError("Pass either a validated request or keyword options, not both.")
            return request
        return SqlPartitionedLoadRequest.model_validate(options)

    # Not a copy-pasted twin: aplan() is already a thin asyncio.to_thread()
    # wrapper delegating to this function via the shared _coerce_request().
    # spaghetti-ignore[sync-async-duplication]: see above
    def plan(
        self,
        request: SqlPartitionedLoadRequest | None = None,
        **options: Any,
    ) -> SqlPartitionPlan:
        return self.plan_request(self._coerce_request(request, options))

    async def aplan(
        self,
        request: SqlPartitionedLoadRequest | None = None,
        **options: Any,
    ) -> SqlPartitionPlan:
        validated_request = self._coerce_request(request, options)
        return await asyncio.to_thread(self.plan_request, validated_request)

    # Not a copy-pasted twin: aload() is already a thin asyncio.to_thread()
    # wrapper delegating to this function via the shared _coerce_request().
    # spaghetti-ignore[sync-async-duplication]: see above
    def load(
        self,
        request: SqlPartitionedLoadRequest | None = None,
        **options: Any,
    ) -> pd.DataFrame | dd.DataFrame:
        return self.load_request(self._coerce_request(request, options))

    async def aload(
        self,
        request: SqlPartitionedLoadRequest | None = None,
        **options: Any,
    ) -> pd.DataFrame | dd.DataFrame:
        validated_request = self._coerce_request(request, options)
        return await asyncio.to_thread(self.load_request, validated_request)

    def plan_request(self, request: SqlPartitionedLoadRequest) -> SqlPartitionPlan:
        plan = self._planner.plan_request(request)
        if request.diagnostics:
            self._log_plan_summary(plan, request)
        return plan

    def _try_fast_path(
        self, request: SqlPartitionedLoadRequest, started: float
    ) -> pd.DataFrame | dd.DataFrame | None:
        """Bypass the planner for small, non-arrow, non-pandas requests.

        Returns None to signal "fall through to full planning" when the probe
        fetch reveals more rows than fit in a single chunk.
        """
        t_prepare = perf_counter()
        prepared_stmt = self._planner.prepare_statement(request)
        probe_limit = request.chunk_size + 1
        if request.limit is not None:
            probe_limit = min(request.limit, probe_limit)
        prepared_stmt = prepared_stmt.limit(probe_limit)
        t_fetch = perf_counter()
        with self.resource.engine.connect() as conn:
            df = pd.read_sql(prepared_stmt, conn)
        t_coerce = perf_counter()
        if len(df) > request.chunk_size:
            return None
        df = SqlPartitionExecutor.align_and_coerce_partition(
            df, self._planner.infer_meta_dtypes(prepared_stmt)
        )
        t_dask = perf_counter()
        result: pd.DataFrame | dd.DataFrame = dd.from_pandas(df, npartitions=1)
        t_end = perf_counter()
        if request.diagnostics:
            self.logger.info(
                "Partitioned SQL fast path: single partition "
                f"rows={len(df)} chunk_size={request.chunk_size} "
                f"total={t_end - started:.3f}s "
                f"prepare_stmt={t_fetch - t_prepare:.3f}s "
                f"db_fetch={t_coerce - t_fetch:.3f}s "
                f"coerce={t_dask - t_coerce:.3f}s "
                f"from_pandas={t_end - t_dask:.3f}s"
            )
        return result

    def _load_via_plan(
        self, request: SqlPartitionedLoadRequest, started: float
    ) -> pd.DataFrame | dd.DataFrame:
        plan = self.plan_request(request)
        # Use the request's use_arrow flag; if the executor was created with a different
        # value, we need to create a new executor with the correct flag.
        executor = self._executor
        if request.use_arrow != self._use_arrow:
            executor = SqlPartitionExecutor(
                self._worker_config,
                self._gate_key,
                use_arrow=request.use_arrow,
            )

        t_plan_fetch = perf_counter()
        prepared_stmt = self._planner.prepare_statement(request)
        t_load = perf_counter()
        result = executor.load_plan(
            plan,
            as_pandas=request.as_pandas,
            max_concurrent_fetches=request.max_concurrent_fetches,
            fetch_partition=self._fetch_partition_static,
            statement=prepared_stmt,
        )
        t_end = perf_counter()
        if request.diagnostics:
            result_partitions = result.npartitions if isinstance(result, dd.DataFrame) else 1
            self.logger.info(
                "Partitioned SQL load completed "
                f"strategy={plan.strategy} partitions={len(plan.partitions)} "
                f"rows={plan.total_rows} result_partitions={result_partitions} "
                f"elapsed={t_end - started:.2f}s "
                f"prepare_stmt={t_load - t_plan_fetch:.3f}s "
                f"load_plan={t_end - t_load:.3f}s"
            )
        return result

    def load_request(self, request: SqlPartitionedLoadRequest) -> pd.DataFrame | dd.DataFrame:
        started = perf_counter()
        # Fast path: bypass planner entirely for small queries
        if not request.use_arrow and not request.as_pandas:
            fast_result = self._try_fast_path(request, started)
            if fast_result is not None:
                return fast_result
        return self._load_via_plan(request, started)

    _fetch_partition_static = staticmethod(SqlPartitionExecutor.fetch_partition)

    def _log_plan_summary(
        self,
        plan: SqlPartitionPlan,
        request: SqlPartitionedLoadRequest,
    ) -> None:
        if hasattr(self.logger, "set_level"):
            self.logger.set_level(Logger.INFO)
        self.logger.info(
            "Partitioned SQL plan "
            f"strategy={plan.strategy} partitions={len(plan.partitions)} rows={plan.total_rows} "
            f"chunk_size={request.chunk_size} max_concurrent_fetches={request.max_concurrent_fetches} "
            f"use_arrow={request.use_arrow}"
        )
