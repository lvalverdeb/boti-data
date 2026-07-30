"""Strategy for the ``sqlalchemy`` backend.

Split out of _backend_strategies.py purely for line-count headroom.
Registered back into that module's registry at the bottom of
_backend_strategies.py to avoid a circular import (this module needs
``BackendStrategy``/``StructuredLoadContext``/``ConfiguredLoadContext``
already defined there).

Calls to ``load_sql``/``load_sql_partitioned``/``AsyncSqlDatabaseResource``
go through the ``_backend_strategies`` module object (not a direct name
import) because tests monkeypatch those three names on
``boti_data.gateway._backend_strategies`` directly — routing through the
module object means the patched value is picked up at call time regardless
of where the calling code itself lives.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import SecretStr
from sqlalchemy.engine import url as sqlalchemy_url
from sqlalchemy.exc import SQLAlchemyError

from boti_data.db import SqlDatabaseConfig, SqlDatabaseResource, SqlPartitionedLoadRequest
from boti_data.field_map import FieldMap

from . import _backend_strategies, _payloads, sql_strategy_configured_async
from ._backend_strategies import BackendStrategy, ConfiguredLoadContext, StructuredLoadContext
from .frame_strategies import FrameResult
from .loaders import build_sql_partitioned_request, read_sql_async
from .normalization import build_partitioned_load_options
from .planning import InChunkPolicy
from .requests import BackendConfig, BackendName, BackendResource, SqlLoadRequest, TableDescription
from .sql_describe import describe_table
from .sql_partitioned_exec import run_async_partitioned_request
from .sql_size_estimation import (
    _DEFAULT_PARTITION_CHUNK_SIZE,
    estimate_configured,
    estimate_configured_async,
    estimate_structured,
    estimate_structured_async,
)

if TYPE_CHECKING:
    from .core import DataGateway

_log = logging.getLogger(__name__)


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
            raise ValueError(
                "from_config requires 'connection_url' for the 'sqlalchemy' backend "
                "(the default used when no 'backend' key is present in config). "
                "If you intended a different backend, pass an explicit 'backend' key "
                "(e.g. 'parquet', 'datacube')."
            )
        connection_url = SecretStr(raw_url) if isinstance(raw_url, str) else raw_url
        return SqlDatabaseConfig(connection_url=connection_url, **cfg)

    # -- Resource construction -----------------------------------------------

    def build_resource(
        self,
        config: BackendConfig,
        *,
        fs: Any | None = None,
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
            return _backend_strategies.load_sql_partitioned(
                ctx.config,
                ctx.resource,
                build_sql_partitioned_request(ctx.opts),
            )

        df_local = _backend_strategies.load_sql(
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
                    async with _backend_strategies.AsyncSqlDatabaseResource(ctx.config) as _tmp:
                        return await run_async_partitioned_request(ctx, _tmp, _req)

                return await _run_with_temp()
            return await run_async_partitioned_request(ctx, async_resource, request)
        assert isinstance(ctx.resource, SqlDatabaseResource)
        return await asyncio.to_thread(
            _backend_strategies.load_sql_partitioned,
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
                async with _backend_strategies.AsyncSqlDatabaseResource(
                    ctx.config
                ) as temp_resource:
                    return await read_sql_async(temp_resource, request)
            except SQLAlchemyError:
                if ctx.resource is None:
                    raise
                assert isinstance(ctx.resource, SqlDatabaseResource)
                return await asyncio.to_thread(_backend_strategies.load_sql, ctx.resource, request)
        return await read_sql_async(async_resource, request)

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
            df = _backend_strategies.load_sql_partitioned(
                ctx.config,
                ctx.resource,
                build_sql_partitioned_request(options_dict),
            )
        else:
            df = _backend_strategies.load_sql(
                ctx.resource,
                self._trusted_sql_request(ctx, statement=stmt, model=model),
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
            # load_configured_sync() is pure blocking SQLAlchemy work; run the
            # whole thing off-thread rather than re-implementing it here.
            return await asyncio.to_thread(self.load_configured_sync, ctx)
        return await self._aload_configured_async_sql(ctx)

    async def _aload_configured_async_sql(
        self,
        ctx: ConfiguredLoadContext,
    ) -> FrameResult:
        return await sql_strategy_configured_async.aload_configured_async_sql(ctx)

    async def _aload_configured_execute_async_resource(
        self,
        ctx: ConfiguredLoadContext,
        resource: Any,
    ) -> FrameResult:
        return await sql_strategy_configured_async.aload_configured_execute_async_resource(
            ctx, resource
        )

    @staticmethod
    def _trusted_sql_request(
        ctx: ConfiguredLoadContext, *, statement: Any, model: Any
    ) -> SqlLoadRequest:
        return sql_strategy_configured_async.trusted_sql_request(
            ctx, statement=statement, model=model
        )

    # -- Async lifecycle -----------------------------------------------------

    async def setup_async_context(self, gateway: DataGateway) -> None:
        assert isinstance(gateway.config, SqlDatabaseConfig)
        parsed = sqlalchemy_url.make_url(gateway.config.connection_url.get_secret_value())
        if parsed.get_dialect().is_async:
            gateway._async_sql_resource = _backend_strategies.AsyncSqlDatabaseResource(
                gateway.config
            )
            await gateway._async_sql_resource.__aenter__()
            gateway._auto_resolver.update_async_resource(gateway._async_sql_resource)
            gateway._configured_loader.update_async_resource(gateway._async_sql_resource)
            gateway._load_executor.update_async_resource(gateway._async_sql_resource)
            gateway._post_processor.set_logger(gateway.logger)

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

    # Not a copy-pasted twin: both are 3-line isinstance dispatchers to
    # already-shared plain functions in sql_size_estimation.py; the real
    # dedup already happened one layer down.
    # spaghetti-ignore[sync-async-duplication]: see above
    def estimate_result_size(
        self,
        ctx: StructuredLoadContext | ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        if isinstance(ctx, ConfiguredLoadContext):
            return estimate_configured(ctx)
        return estimate_structured(ctx)

    async def estimate_result_size_async(
        self,
        ctx: StructuredLoadContext | ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        if isinstance(ctx, ConfiguredLoadContext):
            if ctx.resource is not None:
                return await asyncio.to_thread(self.estimate_result_size, ctx)
            return await estimate_configured_async(ctx)
        return await estimate_structured_async(ctx)

    # -- Schema/row-count discovery ------------------------------------------

    def describe(
        self,
        resource: BackendResource | None,
        table: str,
        *,
        row_count_limit: int,
    ) -> TableDescription:
        if resource is None:
            raise ValueError("describe() requires an active SQL resource.")
        return describe_table(resource, table, row_count_limit=row_count_limit)

    # describe() is already a single quick, bounded query -- offloading it to
    # a thread (like estimate_result_size_async's sync-resource branch above)
    # is enough; no separate async reflection/count path is worth the
    # duplication for something this cheap.
    # spaghetti-ignore[sync-async-duplication]: see above
    async def describe_async(
        self,
        resource: BackendResource | None,
        table: str,
        *,
        row_count_limit: int,
    ) -> TableDescription:
        return await asyncio.to_thread(
            self.describe, resource, table, row_count_limit=row_count_limit
        )
