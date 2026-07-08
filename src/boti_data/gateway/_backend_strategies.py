"""Backend strategy registry and implementations.

Each backend (sqlalchemy, parquet, datacube) is encapsulated in a class
that implements :class:`BackendStrategy`.  Adding a new backend means
creating a new strategy class, implementing the abstract methods, and
registering it — no changes to dispatch chains in *core.py*,
*configured_load.py*, or *return_type.py*.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import dask.dataframe as dd
import fsspec
import pandas as pd
from pydantic import SecretStr
from sqlalchemy.engine import url as sqlalchemy_url
from sqlalchemy.exc import SQLAlchemyError

from boti_data.datacube import DatacubeConfig, DatacubeResource
from boti_data.db import (
    AsyncSqlDatabaseResource,
    SqlDatabaseConfig,
    SqlDatabaseResource,
    SqlPartitionedLoadRequest,
)
from boti_data.db.partitioned_execution import SqlPartitionExecutor
from boti_data.db.partitioned_planner import SqlPartitionPlanner
from boti_data.db.sql_config import WorkerSqlConfig
from boti_data.db.sql_engine import _get_worker_engine_identity
from boti_data.field_map import FieldMap
from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource

from . import _payloads
from .frame_strategies import FrameResult, get_frame_strategy
from .load_request import GatewayLoadRequest
from .loaders import (
    build_sql_partitioned_request,
    load_parquet,
    load_sql,
    load_sql_partitioned,
    read_sql_async,
)
from .normalization import (
    build_partitioned_load_options,
)
from .planning import InChunkPolicy
from .post_process import PostProcessor
from .requests import (
    BackendConfig,
    BackendName,
    BackendResource,
    ParquetLoadRequest,
    ResolvedExecutionMode,
    ReturnType,
    SqlLoadRequest,
)

_DEFAULT_PARTITION_CHUNK_SIZE: int = SqlPartitionedLoadRequest.model_fields["chunk_size"].default

if TYPE_CHECKING:
    from .core import DataGateway


# ---------------------------------------------------------------------------
# Context dataclasses — single-parameter wrappers against parameter bloat
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredLoadContext:
    """Everything a strategy needs to execute a structured-mode load."""

    resource: BackendResource | None
    config: BackendConfig
    opts: dict[str, Any]
    loader_return_type: Literal["pandas", "arrow", "dask"]
    resolved_execution_mode: ResolvedExecutionMode | None = None
    timeout: float | None = None
    post_processor: PostProcessor | None = None
    async_sql_resource: Any | None = None
    request: GatewayLoadRequest | None = None


@dataclass(frozen=True)
class ConfiguredLoadContext:
    """Everything a strategy needs to execute a configured-mode load."""

    resource: BackendResource | None
    config: BackendConfig
    control: GatewayLoadRequest
    combined_filters: dict[str, Any]
    db_filters: dict[str, Any]
    db_columns: list[str] | None
    configured_fieldnames: tuple[str, ...] | None
    field_map: FieldMap
    exclude: bool
    sticky_filters: dict[str, Any]
    return_type: ReturnType
    loader_return_type: Literal["pandas", "arrow", "dask"]
    loader_as_pandas: bool
    execution_mode: ResolvedExecutionMode
    post_processor: PostProcessor
    get_configured_select: Callable[[list[str] | None], tuple[Any, Any]] | None = None
    get_configured_select_async: (
        Callable[[Any, list[str] | None], Awaitable[tuple[Any, Any]]] | None
    ) = None
    async_sql_resource: Any | None = None
    chunk_size: int | None = None


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class BackendStrategy(ABC):
    """Interface that every backend must implement.

    Subclasses are **stateless** — a single singleton instance is shared
    across all :class:`~boti_data.gateway.core.DataGateway` instances.
    """

    @property
    @abstractmethod
    def name(self) -> BackendName:
        """Machine-readable backend identifier (e.g. ``"sqlalchemy"``)."""

    # -- Config construction -------------------------------------------------

    @abstractmethod
    def build_config(self, **kwargs: Any) -> BackendConfig:
        """Build a config object from keyword arguments (``from_backend``)."""

    @abstractmethod
    def build_config_from_dict(self, cfg: dict[str, Any]) -> BackendConfig:
        """Build a config object from a legacy-style dict (``from_config``).

        *cfg* has already had common gateway options (table, field_map, fs, …)
        removed by :func:`~._factory.extract_config_common_options`.
        """

    # -- Resource construction -----------------------------------------------

    @abstractmethod
    def build_resource(
        self,
        config: BackendConfig,
        *,
        fs: fsspec.AbstractFileSystem | None = None,
        fs_factory: Any | None = None,
    ) -> tuple[BackendName, BackendResource | None]:
        """Build the runtime resource for *config*.

        Returns ``(backend_name, resource_or_None)``.
        """

    # -- Config validation ---------------------------------------------------

    def validate_requirements(self, config: BackendConfig, **flags: Any) -> None:
        """Backend-specific config validation (no-op by default)."""

    # -- Structured-mode loads ------------------------------------------------

    @abstractmethod
    def load_structured_sync(self, ctx: StructuredLoadContext) -> FrameResult:
        """Execute a synchronous structured-mode load."""

    @abstractmethod
    async def load_structured_async(self, ctx: StructuredLoadContext) -> FrameResult:
        """Execute an asynchronous structured-mode load."""

    # -- Configured-mode loads -----------------------------------------------

    @abstractmethod
    def load_configured_sync(self, ctx: ConfiguredLoadContext) -> FrameResult:
        """Execute a synchronous configured-mode load."""

    @abstractmethod
    async def load_configured_async(self, ctx: ConfiguredLoadContext) -> FrameResult:
        """Execute an asynchronous configured-mode load."""

    # -- Async lifecycle hooks -----------------------------------------------

    async def setup_async_context(self, gateway: DataGateway) -> None:
        """Async context setup (e.g. opening an async SQL connection)."""

    async def teardown_async_context(self, gateway: DataGateway) -> None:
        """Async context teardown."""

    # -- Options preparation -------------------------------------------------

    def prepare_structured_options(
        self,
        options: dict[str, Any],
        field_map: FieldMap,
        configured: bool,
    ) -> dict[str, Any]:
        """Modify loader options before dispatch (e.g. field-map projection).

        No-op by default — only SQL overrides this.
        """
        return options

    # -- Chunking metadata ---------------------------------------------------

    def supports_chunking(self) -> bool:
        """Whether the backend supports IN-chunk fan-out."""
        return True

    def supports_in_chunk_hinting(self) -> bool:
        """Whether the backend uses SQL-style chunk hints."""
        return False

    def chunk_hint(
        self,
        filters: dict[str, Any],
        policy: InChunkPolicy,
    ) -> dict[str, Any] | None:
        """Return a chunking hint or ``None``.

        Only SQL overrides this.
        """
        return None

    # -- Auto return type ----------------------------------------------------

    def estimate_result_size(
        self,
        ctx: StructuredLoadContext | ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        """Return ``(estimated_rows, estimated_bytes)`` or ``None``.

        Used by :class:`~.return_type.AutoReturnTypeResolver` to decide
        whether a result should be eager (pandas) or lazy (dask).
        Returning ``None`` means "unknown — use the default."
        """
        return None

    async def estimate_result_size_async(
        self,
        ctx: StructuredLoadContext | ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        """Async variant of :meth:`estimate_result_size`.

        Default implementation delegates to the sync method.
        """
        return self.estimate_result_size(ctx)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STRATEGIES_BY_NAME: dict[str, BackendStrategy] = {}
_STRATEGIES_BY_CONFIG_TYPE: dict[type, BackendStrategy] = {}


def register(
    name: str,
    strategy: BackendStrategy,
    *config_types: type,
) -> None:
    """Register a strategy under *name* for the given *config_types*."""
    _STRATEGIES_BY_NAME[name] = strategy
    for ct in config_types:
        _STRATEGIES_BY_CONFIG_TYPE[ct] = strategy


def get(name: str) -> BackendStrategy:
    """Look up a strategy by name."""
    try:
        return _STRATEGIES_BY_NAME[name]
    except KeyError:
        raise ValueError(f"Unknown backend: {name!r}") from None


def for_config(config: BackendConfig) -> BackendStrategy:
    """Look up a strategy by config type."""
    strategy = _STRATEGIES_BY_CONFIG_TYPE.get(type(config))
    if strategy is None:
        raise ValueError(f"No strategy registered for {type(config).__name__}")
    return strategy


# ---------------------------------------------------------------------------
# SqlAlchemyStrategy
# ---------------------------------------------------------------------------


class SqlAlchemyStrategy(BackendStrategy):
    """Strategy for the ``sqlalchemy`` backend."""

    @property
    def name(self) -> BackendName:
        return "sqlalchemy"

    # -- Config construction -------------------------------------------------

    def build_config(self, **kwargs: Any) -> SqlDatabaseConfig:
        return SqlDatabaseConfig(**kwargs)

    def build_config_from_dict(self, cfg: dict[str, Any]) -> SqlDatabaseConfig:
        cfg = dict(cfg)
        raw_url = cfg.pop("connection_url", None)
        if raw_url is None:
            raise ValueError("from_config requires 'connection_url'.")
        connection_url = SecretStr(raw_url) if isinstance(raw_url, str) else raw_url
        return SqlDatabaseConfig(connection_url=connection_url, **cfg)

    # -- Resource construction -----------------------------------------------

    def build_resource(
        self,
        config: BackendConfig,
        *,
        fs: fsspec.AbstractFileSystem | None = None,
        fs_factory: Any | None = None,
    ) -> tuple[BackendName, BackendResource | None]:
        assert isinstance(config, SqlDatabaseConfig)
        if self._is_async_dsn(config):
            return "sqlalchemy", None
        return "sqlalchemy", SqlDatabaseResource(config)

    @staticmethod
    def _is_async_dsn(config: SqlDatabaseConfig) -> bool:
        parsed = sqlalchemy_url.make_url(config.connection_url.get_secret_value())
        return bool(parsed.get_dialect().is_async)

    # -- Options preparation -------------------------------------------------

    def prepare_structured_options(
        self,
        options: dict[str, Any],
        field_map: FieldMap,
        configured: bool,
    ) -> dict[str, Any]:
        if configured:
            return options
        return self._project_through_field_map(options, field_map)

    @staticmethod
    def _project_through_field_map(
        options: dict[str, Any],
        field_map: FieldMap,
    ) -> dict[str, Any]:
        columns = options.get("columns")
        statement = options.get("statement")
        model = options.get("model")
        if not columns or statement is None or model is None:
            return options
        projected_names = [field_map.to_db(column) if field_map else column for column in columns]
        projected_columns = [getattr(model, column) for column in projected_names]
        prepared = dict(options)
        prepared["statement"] = statement.with_only_columns(
            *projected_columns,
            maintain_column_froms=True,
        )
        prepared.pop("columns", None)
        return prepared

    # -- Structured mode loads -----------------------------------------------

    def load_structured_sync(self, ctx: StructuredLoadContext) -> FrameResult:
        assert isinstance(ctx.config, SqlDatabaseConfig)
        if ctx.resource is None:
            raise RuntimeError(
                "DataGateway.load() requires a synchronous DSN. "
                "Switch to a sync DSN (e.g. 'mysql+pymysql://...') or use aload() instead."
            )
        assert isinstance(ctx.resource, SqlDatabaseResource)
        assert ctx.post_processor is not None

        opts_or_request: Any = ctx.request if ctx.request is not None else ctx.opts

        if ctx.resolved_execution_mode == "lazy":
            # Use ctx.opts (already prepared by _project_through_field_map,
            # which projects columns into the statement) — NOT the original
            # request model which has an unprojected statement.
            return load_sql_partitioned(
                ctx.config,
                ctx.resource,
                build_sql_partitioned_request(ctx.opts),
            )

        df_local = load_sql(
            ctx.resource,
            SqlLoadRequest.model_validate(
                _payloads.structured_sql_request_payload(
                    opts_or_request,
                    return_type=ctx.loader_return_type,
                )
            ),
        )
        statement = ctx.opts.get("statement")
        if isinstance(df_local, pd.DataFrame) and statement is not None:
            return ctx.post_processor.coerce_eager_sql_frame(
                df_local,
                statement=statement,
            )
        return df_local

    async def load_structured_async(self, ctx: StructuredLoadContext) -> FrameResult:
        assert isinstance(ctx.config, SqlDatabaseConfig)
        assert ctx.post_processor is not None

        if ctx.resolved_execution_mode == "lazy":
            return await self._aload_execute_sql_lazy(ctx)
        return await self._aload_execute_sql_eager(ctx)

    async def _aload_execute_sql_eager(
        self,
        ctx: StructuredLoadContext,
    ) -> FrameResult:
        opts_or_request: Any = ctx.request if ctx.request is not None else ctx.opts
        request = SqlLoadRequest.model_validate(
            _payloads.structured_sql_request_payload(
                opts_or_request,
                return_type=ctx.loader_return_type,
            )
        )
        frame = await self._aload_sql(ctx, request)
        statement = ctx.request.statement if ctx.request is not None else ctx.opts.get("statement")
        if statement is not None and isinstance(frame, pd.DataFrame):
            assert ctx.post_processor is not None
            frame = ctx.post_processor.coerce_eager_sql_frame(
                frame,
                statement=statement,
            )
        return frame

    async def _aload_execute_sql_lazy(
        self,
        ctx: StructuredLoadContext,
    ) -> FrameResult:
        # Use ctx.opts (already prepared by _project_through_field_map,
        # which projects columns into the statement) — NOT the original
        # request model which has an unprojected statement.
        request = build_sql_partitioned_request(ctx.opts)
        if ctx.resource is None:
            async_resource = ctx.async_sql_resource
            if async_resource is None:
                assert isinstance(ctx.config, SqlDatabaseConfig)

                async def _run_with_temp(
                    _req: SqlPartitionedLoadRequest = request,
                ) -> FrameResult:
                    async with AsyncSqlDatabaseResource(ctx.config) as _tmp:
                        return await self._run_async_partitioned_request(ctx, _tmp, _req)

                return await _run_with_temp()
            return await self._run_async_partitioned_request(ctx, async_resource, request)
        assert isinstance(ctx.resource, SqlDatabaseResource)
        return await asyncio.to_thread(
            load_sql_partitioned,
            ctx.config,
            ctx.resource,
            request,
        )

    async def _aload_sql(
        self,
        ctx: StructuredLoadContext,
        request: SqlLoadRequest,
    ) -> FrameResult:
        async_resource = ctx.async_sql_resource
        if async_resource is None:
            assert isinstance(ctx.config, SqlDatabaseConfig)
            try:
                async with AsyncSqlDatabaseResource(ctx.config) as temp_resource:
                    return await read_sql_async(temp_resource, request)
            except SQLAlchemyError:
                if ctx.resource is None:
                    raise
                assert isinstance(ctx.resource, SqlDatabaseResource)
                return await asyncio.to_thread(load_sql, ctx.resource, request)
        return await read_sql_async(async_resource, request)

    @staticmethod
    async def _run_async_partitioned_request(
        ctx: StructuredLoadContext,
        async_resource: Any,
        request: SqlPartitionedLoadRequest,
    ) -> FrameResult:
        assert isinstance(ctx.config, SqlDatabaseConfig)

        class _EngineAdapter:
            engine = async_resource.engine.sync_engine
            logger = async_resource.logger
            debug = False

        started = perf_counter()
        t_planner_created = perf_counter()
        planner = SqlPartitionPlanner(_EngineAdapter())  # type: ignore[arg-type]

        # Fast path: bypass planner entirely for small queries
        # Fetch LIMIT chunk_size + 1 rows directly; if <= chunk_size, return immediately.
        # Eliminates the COUNT query overhead for the common single-partition case.
        if not request.use_arrow and not request.as_pandas:
            t_prepare = perf_counter()
            prepared_stmt = planner.prepare_statement(request)
            if request.limit is not None:
                prepared_stmt = prepared_stmt.limit(request.limit)
            t_fetch = perf_counter()
            async with async_resource.engine.connect() as conn:
                df = await conn.run_sync(
                    lambda sync_conn: pd.read_sql(prepared_stmt, sync_conn),
                )
            t_coerce = perf_counter()
            if len(df) <= request.chunk_size:
                df = SqlPartitionExecutor.align_and_coerce_partition(
                    df, planner.infer_meta_dtypes(prepared_stmt),
                )
                t_dask = perf_counter()
                result = dd.from_pandas(df, npartitions=1)
                t_end = perf_counter()
                if request.diagnostics:
                    async_resource.logger.info(
                        "Partitioned SQL fast path: single partition "
                        f"rows={len(df)} chunk_size={request.chunk_size} "
                        f"total={t_end - started:.3f}s "
                        f"planner_init={t_prepare - t_planner_created:.3f}s "
                        f"prepare_stmt={t_fetch - t_prepare:.3f}s "
                        f"db_fetch={t_coerce - t_fetch:.3f}s "
                        f"coerce={t_dask - t_coerce:.3f}s "
                        f"from_pandas={t_end - t_dask:.3f}s"
                    )
                return result
            if request.diagnostics:
                async_resource.logger.info(
                    "Partitioned SQL fast path fallback: rows exceed chunk_size, "
                    f"rows_fetched={len(df)} chunk_size={request.chunk_size} "
                    f"prepare_stmt={t_fetch - t_prepare:.3f}s "
                    f"db_fetch={perf_counter() - t_fetch:.3f}s"
                )

        t_plan = perf_counter()
        plan = await planner.async_plan_request(request, async_resource.engine)
        t_plan_done = perf_counter()
        if request.diagnostics:
            async_resource.logger.info(
                "Partitioned SQL plan "
                f"strategy={plan.strategy} partitions={len(plan.partitions)} "
                f"rows={plan.total_rows} "
                f"chunk_size={request.chunk_size} "
                f"max_concurrent_fetches={request.max_concurrent_fetches} "
                f"use_arrow={request.use_arrow} "
                f"plan_elapsed={t_plan_done - t_plan:.3f}s"
            )

        worker_config = WorkerSqlConfig.from_database_config(ctx.config)
        gate_key = _get_worker_engine_identity(worker_config)

        if len(plan.partitions) == 1 and not request.use_arrow and not request.as_pandas:
            t_prepare2 = perf_counter()
            prepared_stmt = planner.prepare_statement(request)
            t_fetch2 = perf_counter()
            async with async_resource.engine.connect() as conn:
                df = await conn.run_sync(
                    lambda sync_conn: pd.read_sql(prepared_stmt, sync_conn),
                )
            t_coerce2 = perf_counter()
            df = SqlPartitionExecutor.align_and_coerce_partition(df, plan.meta_dtypes)
            t_dask2 = perf_counter()
            result = dd.from_pandas(df, npartitions=1)
            t_end2 = perf_counter()
            if request.diagnostics:
                async_resource.logger.info(
                    "Partitioned SQL single-partition plan path "
                    f"rows={len(df)} "
                    f"prepare_stmt={t_fetch2 - t_prepare2:.3f}s "
                    f"db_fetch={t_coerce2 - t_fetch2:.3f}s "
                    f"coerce={t_dask2 - t_coerce2:.3f}s "
                    f"from_pandas={t_end2 - t_dask2:.3f}s"
                )
        else:
            executor = SqlPartitionExecutor(
                worker_config,
                gate_key,
                use_arrow=request.use_arrow,
            )
            prepared_stmt = planner.prepare_statement(request)
            result = executor.load_plan(
                plan,
                as_pandas=request.as_pandas,
                max_concurrent_fetches=request.max_concurrent_fetches,
                fetch_partition=SqlPartitionExecutor.fetch_partition_async,
                statement=prepared_stmt,
            )
        if request.diagnostics:
            result_partitions = result.npartitions if isinstance(result, dd.DataFrame) else 1
            async_resource.logger.info(
                "Partitioned SQL load completed "
                f"strategy={plan.strategy} partitions={len(plan.partitions)} "
                f"rows={plan.total_rows} result_partitions={result_partitions} "
                f"elapsed={perf_counter() - started:.2f}s"
            )
        return result

    # -- Configured mode loads -----------------------------------------------

    def load_configured_sync(self, ctx: ConfiguredLoadContext) -> FrameResult:
        assert isinstance(ctx.config, SqlDatabaseConfig)
        assert isinstance(ctx.resource, SqlDatabaseResource)
        assert ctx.get_configured_select is not None

        model, stmt = ctx.get_configured_select(ctx.db_columns)
        if ctx.execution_mode == "lazy":
            partitioned_options = build_partitioned_load_options(
                statement=stmt,
                model=model,
                filters=ctx.db_filters,
                control=ctx.control.model_dump(),
                default_chunk_size=(ctx.chunk_size or _DEFAULT_PARTITION_CHUNK_SIZE),
            )
            options_dict = partitioned_options.model_dump(exclude_none=True)
            options_dict["use_arrow"] = False
            df = load_sql_partitioned(
                ctx.config,
                ctx.resource,
                build_sql_partitioned_request(options_dict),
            )
        else:
            df = load_sql(
                ctx.resource,
                self._trusted_sql_request(
                    statement=stmt,
                    model=model,
                    filters=ctx.db_filters,
                    params=ctx.control.params,
                    limit=ctx.control.limit,
                    as_pandas=True,
                    diagnostics=ctx.control.diagnostics,
                    return_type=ctx.loader_return_type,
                ),
            )
            if isinstance(df, pd.DataFrame):
                df = ctx.post_processor.coerce_eager_sql_frame(df, statement=stmt)

        return ctx.post_processor.finalize_configured_result(
            df,
            return_type=ctx.return_type,
            apply_field_map=True,
            fieldnames=ctx.configured_fieldnames,
        )

    async def load_configured_async(self, ctx: ConfiguredLoadContext) -> FrameResult:
        assert isinstance(ctx.config, SqlDatabaseConfig)

        if ctx.resource is not None:
            return await self._aload_configured_sync_sql(ctx)
        return await self._aload_configured_async_sql(ctx)

    async def _aload_configured_sync_sql(
        self,
        ctx: ConfiguredLoadContext,
    ) -> FrameResult:
        assert isinstance(ctx.resource, SqlDatabaseResource)
        assert ctx.get_configured_select is not None

        model, stmt = ctx.get_configured_select(ctx.db_columns)
        partitioned_options = build_partitioned_load_options(
            statement=stmt,
            model=model,
            filters=ctx.db_filters,
            control=ctx.control.model_dump(),
            default_chunk_size=(ctx.chunk_size or _DEFAULT_PARTITION_CHUNK_SIZE),
        )
        if ctx.execution_mode == "lazy":
            options_dict = partitioned_options.model_dump(exclude_none=True)
            options_dict["use_arrow"] = False
            request = build_sql_partitioned_request(options_dict)
            df = await asyncio.to_thread(
                load_sql_partitioned,
                ctx.config,
                ctx.resource,
                request,
            )
        else:
            df = await asyncio.to_thread(
                load_sql,
                ctx.resource,
                self._trusted_sql_request(
                    statement=stmt,
                    model=model,
                    filters=ctx.db_filters,
                    params=ctx.control.params,
                    limit=ctx.control.limit,
                    as_pandas=True,
                    diagnostics=ctx.control.diagnostics,
                    return_type=ctx.loader_return_type,
                ),
            )
            if isinstance(df, pd.DataFrame):
                df = ctx.post_processor.coerce_eager_sql_frame(df, statement=stmt)

        return ctx.post_processor.finalize_configured_result(
            df,
            return_type=ctx.return_type,
            apply_field_map=True,
            fieldnames=ctx.configured_fieldnames,
        )

    async def _aload_configured_async_sql(
        self,
        ctx: ConfiguredLoadContext,
    ) -> FrameResult:
        assert ctx.get_configured_select_async is not None

        async def _execute(resource: Any) -> FrameResult:
            return await self._aload_configured_execute_async_resource(
                ctx,
                resource,
            )

        async_resource = ctx.async_sql_resource
        if async_resource is None:
            async with AsyncSqlDatabaseResource(ctx.config) as resource:
                df = await _execute(resource)
        else:
            df = await _execute(async_resource)

        return ctx.post_processor.finalize_configured_result(
            df,
            return_type=ctx.return_type,
            apply_field_map=True,
            fieldnames=ctx.configured_fieldnames,
        )

    async def _aload_configured_execute_async_resource(
        self,
        ctx: ConfiguredLoadContext,
        resource: Any,
    ) -> FrameResult:
        diagnostics = ctx.control.diagnostics
        assert ctx.get_configured_select_async is not None
        t0 = perf_counter()
        model, stmt = await ctx.get_configured_select_async(
            resource,
            ctx.db_columns,
        )
        t1 = perf_counter()
        if diagnostics:
            resource.logger.info(
                "Configured async select "
                f"reflect_select={t1 - t0:.3f}s"
            )
        if ctx.execution_mode == "lazy":
            t2 = perf_counter()
            partitioned_options = build_partitioned_load_options(
                statement=stmt,
                model=model,
                filters=ctx.db_filters,
                control=ctx.control.model_dump(),
                default_chunk_size=(ctx.chunk_size or _DEFAULT_PARTITION_CHUNK_SIZE),
            )
            options_dict = partitioned_options.model_dump(exclude_none=True)
            options_dict["use_arrow"] = False
            request = build_sql_partitioned_request(options_dict)
            t3 = perf_counter()
            if diagnostics:
                resource.logger.info(
                    "Build partitioned request "
                    f"elapsed={t3 - t2:.3f}s"
                )
            return await self._run_async_partitioned_request(ctx, resource, request)

        df = await read_sql_async(
            resource,
            self._trusted_sql_request(
                statement=stmt,
                model=model,
                filters=ctx.db_filters,
                params=ctx.control.params,
                limit=ctx.control.limit,
                as_pandas=True,
                diagnostics=ctx.control.diagnostics,
                return_type=ctx.loader_return_type,
            ),
        )
        if isinstance(df, pd.DataFrame):
            df = ctx.post_processor.coerce_eager_sql_frame(df, statement=stmt)
        return df

    @staticmethod
    def _trusted_sql_request(
        *,
        sql: str | None = None,
        statement: Any = None,
        model: Any = None,
        filters: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        limit: int | None = None,
        columns: list[str] | None = None,
        as_pandas: bool = False,
        diagnostics: bool = False,
        return_type: str = "pandas",
    ) -> SqlLoadRequest:
        return SqlLoadRequest.model_construct(
            sql=sql,
            statement=statement,
            model=model,
            filters=filters or {},
            params=params or {},
            limit=limit,
            columns=columns,
            as_pandas=as_pandas,
            diagnostics=diagnostics,
            return_type=return_type,
        )

    # -- Async lifecycle -----------------------------------------------------

    async def setup_async_context(self, gateway: DataGateway) -> None:
        assert isinstance(gateway.config, SqlDatabaseConfig)
        parsed = sqlalchemy_url.make_url(gateway.config.connection_url.get_secret_value())
        if parsed.get_dialect().is_async:
            gateway._async_sql_resource = AsyncSqlDatabaseResource(gateway.config)
            await gateway._async_sql_resource.__aenter__()
            gateway._auto_resolver.update_async_resource(gateway._async_sql_resource)
            gateway._configured_loader.update_async_resource(gateway._async_sql_resource)
            gateway._post_processor.set_logger(gateway._logger)

    async def teardown_async_context(self, gateway: DataGateway) -> None:
        if gateway._async_sql_resource is not None:
            await gateway._async_sql_resource.__aexit__(None, None, None)
            gateway._async_sql_resource = None

    # -- Chunking ------------------------------------------------------------

    def supports_in_chunk_hinting(self) -> bool:
        return True

    def chunk_hint(
        self,
        filters: dict[str, Any],
        policy: InChunkPolicy,
    ) -> dict[str, Any] | None:
        from boti_data.filters import FilterHandler

        return FilterHandler("sqlalchemy").suggest_sql_in_chunking(
            filters,
            chunk_size=policy.eager_auto_min_values,
            max_concurrency=policy.eager_auto_concurrency or 8,
        )

    # -- Auto return type ----------------------------------------------------

    def estimate_result_size(
        self,
        ctx: StructuredLoadContext | ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        if isinstance(ctx, ConfiguredLoadContext):
            return self._estimate_configured(ctx)
        return self._estimate_structured(ctx)

    @staticmethod
    def _estimate_structured(
        ctx: StructuredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        from boti_data.db.partitioned_planner import SqlPartitionPlanner

        options = ctx.opts
        if options.get("partitioned") is True:
            return None
        if options.get("sql") is not None:
            return None
        if options.get("statement") is None or options.get("model") is None:
            return None
        request = build_sql_partitioned_request(options)
        threshold = _auto_sql_threshold(request.limit)
        if threshold == 0:
            return None
        if not isinstance(ctx.resource, SqlDatabaseResource):
            return None
        planner = SqlPartitionPlanner(ctx.resource)
        statement = planner.prepare_statement(request)
        probed_rows = min(planner.count_rows_up_to(statement, threshold), threshold)
        return probed_rows, None

    @staticmethod
    def _estimate_configured(
        ctx: ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        if ctx.control.partitioned is True:
            return None
        limit = ctx.control.limit
        if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
            return limit, None
        if not isinstance(ctx.resource, SqlDatabaseResource):
            return None
        if ctx.get_configured_select is None:
            return None
        model, stmt = ctx.get_configured_select(ctx.db_columns)
        partitioned_options = build_partitioned_load_options(
            statement=stmt,
            model=model,
            filters=ctx.db_filters,
            control=ctx.control.model_dump(),
            default_chunk_size=ctx.chunk_size or _DEFAULT_PARTITION_CHUNK_SIZE,
        )
        request = build_sql_partitioned_request(partitioned_options.model_dump(exclude_none=True))
        from boti_data.db.partitioned_planner import SqlPartitionPlanner

        planner = SqlPartitionPlanner(ctx.resource)
        statement = planner.prepare_statement(request)
        threshold = _auto_sql_threshold(request.limit)
        if threshold == 0:
            return None
        probed_rows = min(planner.count_rows_up_to(statement, threshold), threshold)
        return probed_rows, None

    async def estimate_result_size_async(
        self,
        ctx: StructuredLoadContext | ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        if isinstance(ctx, ConfiguredLoadContext):
            if ctx.resource is not None:
                return await asyncio.to_thread(self.estimate_result_size, ctx)
            return await self._estimate_configured_async(ctx)
        return await self._estimate_structured_async(ctx)

    async def _estimate_structured_async(
        self,
        ctx: StructuredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        options = ctx.opts
        if options.get("partitioned") is True:
            return None
        if options.get("sql") is not None:
            return None
        if options.get("statement") is None or options.get("model") is None:
            return None
        request = build_sql_partitioned_request(options)
        threshold = _auto_sql_threshold(request.limit)
        if threshold == 0:
            return None
        if ctx.resource is not None:
            assert isinstance(ctx.resource, SqlDatabaseResource)
            planner = SqlPartitionPlanner(ctx.resource)
            statement = planner.prepare_statement(request)
            probed_rows = min(planner.count_rows_up_to(statement, threshold), threshold)
            return probed_rows, None
        return await self._probe_async(request, threshold, ctx.async_sql_resource)

    async def _estimate_configured_async(
        self,
        ctx: ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        if ctx.control.partitioned is True:
            return None
        limit = ctx.control.limit
        if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
            return limit, None
        assert ctx.get_configured_select_async is not None
        model, stmt = await ctx.get_configured_select_async(
            ctx.async_sql_resource,
            ctx.db_columns,
        )
        partitioned_options = build_partitioned_load_options(
            statement=stmt,
            model=model,
            filters=ctx.db_filters,
            control=ctx.control.model_dump(),
            default_chunk_size=ctx.chunk_size or _DEFAULT_PARTITION_CHUNK_SIZE,
        )
        request = build_sql_partitioned_request(partitioned_options.model_dump(exclude_none=True))
        return await self._probe_async(
            request,
            _auto_sql_threshold(request.limit),
            ctx.async_sql_resource,
        )

    @staticmethod
    async def _probe_async(
        request: SqlPartitionedLoadRequest,
        threshold: int,
        async_resource: Any,
    ) -> tuple[int | None, int | None] | None:
        if threshold == 0:
            return None
        if async_resource is None:
            return None
        planner = SqlPartitionPlanner(
            type(
                "_Adapter",
                (),
                {
                    "engine": async_resource.engine.sync_engine,
                    "logger": async_resource.logger,
                    "debug": False,
                },
            )()
        )
        statement = planner.prepare_statement(request)
        async with async_resource.engine.connect() as conn:
            probed_rows = await planner._async_count_rows_up_to(statement, threshold, conn)
        return min(probed_rows, threshold), None


# ---------------------------------------------------------------------------
# ParquetStrategy
# ---------------------------------------------------------------------------


class ParquetStrategy(BackendStrategy):
    """Strategy for the ``parquet`` backend."""

    @property
    def name(self) -> BackendName:
        return "parquet"

    # -- Config construction -------------------------------------------------

    def build_config(self, **kwargs: Any) -> ParquetDataConfig:
        return ParquetDataConfig(**kwargs)

    def build_config_from_dict(self, cfg: dict[str, Any]) -> ParquetDataConfig:
        cfg = dict(cfg)
        storage_path = cfg.pop("storage_path", None)
        if storage_path is not None and "parquet_storage_path" not in cfg:
            cfg["parquet_storage_path"] = storage_path
        if self._should_default_partition_on(cfg):
            cfg["partition_on"] = ["partition_date"]
        return ParquetDataConfig(**cfg)

    @staticmethod
    def _should_default_partition_on(cfg: dict[str, Any]) -> bool:
        return (
            "partition_on" not in cfg
            and cfg.get("parquet_start_date") is not None
            and cfg.get("parquet_end_date") is not None
        )

    # -- Resource construction -----------------------------------------------

    def build_resource(
        self,
        config: BackendConfig,
        *,
        fs: fsspec.AbstractFileSystem | None = None,
        fs_factory: Any | None = None,
    ) -> tuple[BackendName, BackendResource]:
        assert isinstance(config, ParquetDataConfig)
        return "parquet", ParquetDataResource(config, fs=fs, fs_factory=fs_factory)

    # -- Structured mode loads -----------------------------------------------

    def load_structured_sync(self, ctx: StructuredLoadContext) -> FrameResult:
        # Use ctx.opts (which still carries bare runtime filter kwargs) rather
        # than ctx.request: GatewayLoadRequest forbids extra fields, so bare
        # kwargs never land on request.filters and would be silently dropped.
        payload_source: Any = ctx.opts if ctx.opts is not None else ctx.request
        request = ParquetLoadRequest.model_validate(
            _payloads.structured_parquet_request_payload(
                payload_source,
                return_type=ctx.loader_return_type,
            )
        )
        return load_parquet(ctx.resource, request)

    async def load_structured_async(self, ctx: StructuredLoadContext) -> FrameResult:
        # See load_structured_sync: ctx.opts preserves runtime filter values.
        payload_source: Any = ctx.opts if ctx.opts is not None else ctx.request
        request = ParquetLoadRequest.model_validate(
            _payloads.structured_parquet_request_payload(
                payload_source,
                return_type=ctx.loader_return_type,
            )
        )
        return await asyncio.to_thread(load_parquet, ctx.resource, request)

    # -- Configured mode loads -----------------------------------------------

    def load_configured_sync(self, ctx: ConfiguredLoadContext) -> FrameResult:
        assert isinstance(ctx.resource, ParquetDataResource)
        df = load_parquet(
            ctx.resource,
            ParquetLoadRequest.model_construct(
                filters=ctx.combined_filters,
                raw_filters=ctx.control.raw_filters,
                limit=ctx.control.limit,
                columns=(list(ctx.configured_fieldnames) if ctx.configured_fieldnames else None),
                as_pandas=ctx.loader_as_pandas,
                diagnostics=ctx.control.diagnostics,
                return_type=ctx.loader_return_type,
            ),
        )
        return ctx.post_processor.finalize_configured_result(
            df,
            return_type=ctx.return_type,
            apply_field_map=False,
            fieldnames=ctx.configured_fieldnames,
        )

    async def load_configured_async(self, ctx: ConfiguredLoadContext) -> FrameResult:
        assert isinstance(ctx.resource, ParquetDataResource)
        df = await asyncio.to_thread(
            load_parquet,
            ctx.resource,
            ParquetLoadRequest.model_construct(
                filters=ctx.combined_filters,
                raw_filters=ctx.control.raw_filters,
                limit=ctx.control.limit,
                columns=(list(ctx.configured_fieldnames) if ctx.configured_fieldnames else None),
                as_pandas=ctx.loader_as_pandas,
                diagnostics=ctx.control.diagnostics,
                return_type=ctx.loader_return_type,
            ),
        )
        return ctx.post_processor.finalize_configured_result(
            df,
            return_type=ctx.return_type,
            apply_field_map=False,
            fieldnames=ctx.configured_fieldnames,
        )

    # -- Auto return type ----------------------------------------------------

    def estimate_result_size(
        self,
        ctx: StructuredLoadContext | ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        if isinstance(ctx, StructuredLoadContext):
            return self._estimate_structured(ctx)
        return self._estimate_configured(ctx)

    @staticmethod
    def _estimate_structured(
        ctx: StructuredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        options = ctx.opts
        limit = options.get("limit")
        if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
            return limit, None
        resource = ctx.resource
        if not isinstance(resource, ParquetDataResource):
            return None
        try:
            return _resolve_parquet_scan_summary(resource)
        except Exception:
            return None

    @staticmethod
    def _estimate_configured(
        ctx: ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        limit = ctx.control.limit
        if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
            return limit, None
        resource = ctx.resource
        if not isinstance(resource, ParquetDataResource):
            return None
        try:
            return _resolve_parquet_scan_summary(resource)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# DatacubeStrategy
# ---------------------------------------------------------------------------


class DatacubeStrategy(BackendStrategy):
    """Strategy for the ``datacube`` backend."""

    @property
    def name(self) -> BackendName:
        return "datacube"

    # -- Config construction -------------------------------------------------

    def build_config(self, **kwargs: Any) -> DatacubeConfig:
        return DatacubeConfig(**kwargs)

    def build_config_from_dict(self, cfg: dict[str, Any]) -> DatacubeConfig:
        return DatacubeConfig(**cfg)

    # -- Resource construction -----------------------------------------------

    def build_resource(
        self,
        config: BackendConfig,
        *,
        fs: fsspec.AbstractFileSystem | None = None,
        fs_factory: Any | None = None,
    ) -> tuple[BackendName, BackendResource]:
        assert isinstance(config, DatacubeConfig)
        return "datacube", DatacubeResource(config)

    # -- Config validation ---------------------------------------------------

    def validate_requirements(self, config: BackendConfig, **flags: Any) -> None:
        required = flags.get("require_datacube_request_validator", False)
        if not required:
            return
        if not isinstance(config, DatacubeConfig):
            raise ValueError("require_datacube_request_validator=True requires DatacubeConfig.")
        contract = config.contract
        if contract is None or contract.request_validator is None:
            raise ValueError(
                "require_datacube_request_validator=True requires DatacubeContract "
                "with a request_validator."
            )

    # -- Structured mode loads -----------------------------------------------

    def load_structured_sync(self, ctx: StructuredLoadContext) -> FrameResult:
        assert isinstance(ctx.resource, DatacubeResource) or hasattr(ctx.resource, "load")
        # Use ctx.opts (includes runtime filter values) rather than ctx.request
        # (which only has control keys — runtime filters are excluded from the
        # GatewayLoadRequest model by design).
        return ctx.resource.load(_payloads.datacube_request_from_options(ctx.opts))

    async def load_structured_async(self, ctx: StructuredLoadContext) -> FrameResult:
        assert isinstance(ctx.resource, DatacubeResource) or hasattr(ctx.resource, "aload")
        return await ctx.resource.aload(_payloads.datacube_request_from_options(ctx.opts))

    # -- Configured mode loads -----------------------------------------------

    def load_configured_sync(self, ctx: ConfiguredLoadContext) -> FrameResult:
        from boti_data.datacube.contract import DatacubeRequest

        request = DatacubeRequest(
            filters=ctx.combined_filters,
            cube=ctx.control.cube,
        )
        df = ctx.resource.load(request)
        return get_frame_strategy(ctx.return_type).normalize(df)

    async def load_configured_async(self, ctx: ConfiguredLoadContext) -> FrameResult:
        from boti_data.datacube.contract import DatacubeRequest

        request = DatacubeRequest(
            filters=ctx.combined_filters,
            cube=ctx.control.cube,
        )
        df = await ctx.resource.aload(request)
        return get_frame_strategy(ctx.return_type).normalize(df)

    # -- Chunking ------------------------------------------------------------

    def supports_chunking(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Auto-register built-in strategies
# ---------------------------------------------------------------------------

_SQLALCHEMY = SqlAlchemyStrategy()
_PARQUET = ParquetStrategy()
_DATACUBE = DatacubeStrategy()

register("sqlalchemy", _SQLALCHEMY, SqlDatabaseConfig)
register("parquet", _PARQUET, ParquetDataConfig)
register("datacube", _DATACUBE, DatacubeConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AUTO_EAGER_MAX_ROWS = 10_000
_AUTO_EAGER_MAX_FILES = 4
_AUTO_EAGER_MAX_BYTES = 32 * 1024 * 1024


def _auto_sql_threshold(limit: int | None = None) -> int:
    effective_limit = _AUTO_EAGER_MAX_ROWS if limit is None else min(limit, _AUTO_EAGER_MAX_ROWS)
    return max(0, effective_limit)


def _resolve_parquet_scan_summary(
    resource: ParquetDataResource,
) -> tuple[int | None, int | None]:
    files_to_load = resource._resolve_files_to_load()
    if not files_to_load:
        return 0, 0
    if len(files_to_load) > _AUTO_EAGER_MAX_FILES:
        return None, None
    total_bytes = 0
    for path in files_to_load:
        info = resource._get_file_info(path)
        if info is None:
            return None, None
        size = info.get("size")
        if not isinstance(size, int):
            return None, None
        total_bytes += size
    return None, total_bytes
