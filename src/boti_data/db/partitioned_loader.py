from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any, Optional

import dask.dataframe as dd
import pandas as pd
from sqlalchemy.engine import url as sqlalchemy_url

from boti.core.logger import Logger
from boti_data.db.partitioned_execution import SqlPartitionExecutor
from boti_data.db.partitioned_planner import SqlPartitionPlanner
from boti_data.db.partitioned_types import (
    PartitionStrategy,
    SqlPartitionParams,
    SqlPartitionPlan,
    SqlPartitionSpec,
    SqlPartitionedLoadRequest,
)
from boti_data.db.sql_config import SqlDatabaseConfig, WorkerSqlConfig
from boti_data.db.sql_engine import _get_worker_engine_identity
from boti_data.db.sql_resource import SqlDatabaseResource


class SqlPartitionedLoader:
    """Partition-aware SQL loader that returns a lazy Dask DataFrame."""

    def __init__(
        self,
        config: SqlDatabaseConfig,
        *,
        resource: Optional[SqlDatabaseResource] = None,
        use_arrow: bool = True,
    ) -> None:
        self.config = config
        self._resource = resource or SqlDatabaseResource(config)
        self._owns_resource = resource is None
        self._worker_config = WorkerSqlConfig.from_database_config(config)
        self._gate_key = _get_worker_engine_identity(self._worker_config)
        self._planner = SqlPartitionPlanner(self.resource)
        self._executor = SqlPartitionExecutor(
            self._worker_config, self._gate_key, use_arrow=use_arrow,
        )
        self._use_arrow = use_arrow
        self._validate_distributed_config()

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

    def close(self) -> None:
        if self._owns_resource:
            self.resource.close()

    def __enter__(self) -> "SqlPartitionedLoader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @staticmethod
    def _coerce_request(
        request: Optional[SqlPartitionedLoadRequest],
        options: dict[str, Any],
    ) -> SqlPartitionedLoadRequest:
        if request is not None:
            if options:
                raise TypeError("Pass either a validated request or keyword options, not both.")
            return request
        return SqlPartitionedLoadRequest.model_validate(options)

    def plan(
        self,
        request: Optional[SqlPartitionedLoadRequest] = None,
        **options: Any,
    ) -> SqlPartitionPlan:
        return self.plan_request(self._coerce_request(request, options))

    async def aplan(
        self,
        request: Optional[SqlPartitionedLoadRequest] = None,
        **options: Any,
    ) -> SqlPartitionPlan:
        validated_request = self._coerce_request(request, options)
        return await asyncio.to_thread(self.plan_request, validated_request)

    def load(
        self,
        request: Optional[SqlPartitionedLoadRequest] = None,
        **options: Any,
    ) -> pd.DataFrame | dd.DataFrame:
        return self.load_request(self._coerce_request(request, options))

    async def aload(
        self,
        request: Optional[SqlPartitionedLoadRequest] = None,
        **options: Any,
    ) -> pd.DataFrame | dd.DataFrame:
        validated_request = self._coerce_request(request, options)
        return await asyncio.to_thread(self.load_request, validated_request)

    def plan_request(self, request: SqlPartitionedLoadRequest) -> SqlPartitionPlan:
        plan = self._planner.plan_request(request)
        if request.diagnostics:
            self._log_plan_summary(plan, request)
        return plan

    def load_request(self, request: SqlPartitionedLoadRequest) -> pd.DataFrame | dd.DataFrame:
        started = perf_counter()
        plan = self.plan_request(request)
        # Use the request's use_arrow flag; if the executor was created with a different
        # value, we need to create a new executor with the correct flag.
        executor = self._executor
        if request.use_arrow != self._use_arrow:
            executor = SqlPartitionExecutor(
                self._worker_config, self._gate_key, use_arrow=request.use_arrow,
            )

        fetch_fn = self._fetch_partition_static
        result = executor.load_plan(
            plan,
            as_pandas=request.as_pandas,
            max_concurrent_fetches=request.max_concurrent_fetches,
            fetch_partition=fetch_fn,
        )
        if request.diagnostics:
            elapsed = perf_counter() - started
            result_partitions = result.npartitions if isinstance(result, dd.DataFrame) else 1
            self.logger.info(
                "Partitioned SQL load completed "
                f"strategy={plan.strategy} partitions={len(plan.partitions)} "
                f"rows={plan.total_rows} result_partitions={result_partitions} "
                f"elapsed={elapsed:.2f}s"
            )
        return result

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
