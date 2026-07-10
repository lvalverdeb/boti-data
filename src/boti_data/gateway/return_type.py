from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from boti_data.db import AsyncSqlDatabaseResource
from boti_data.field_map import FieldMap

from ._backend_strategies import (
    BackendStrategy,
    ConfiguredLoadContext,
    StructuredLoadContext,
)
from .load_request import GatewayLoadRequest
from .requests import (
    BackendConfig,
    BackendResource,
    ConfiguredRequest,
    DataFrameParams,
    ResolvedReturnType,
)

_AUTO_EAGER_MAX_ROWS = 10_000
_AUTO_EAGER_MAX_BYTES = 32 * 1024 * 1024


class AutoReturnTypeResolver:
    """Resolves ``return_type='auto'`` by probing row counts or file sizes."""

    def __init__(
        self,
        *,
        config: BackendConfig,
        resource: BackendResource | None,
        strategy: BackendStrategy,
        field_map: FieldMap,
        async_sql_resource: AsyncSqlDatabaseResource | None,
        df_params: DataFrameParams,
        configured: bool,
        build_configured_request: Callable[[dict[str, Any]], ConfiguredRequest],
        get_configured_select: Callable[[list[str] | None], tuple[Any, Any]] | None = None,
        get_configured_select_async: Callable[[Any, list[str] | None], Awaitable[tuple[Any, Any]]] | None = None,
        configured_fieldnames: Callable[[dict[str, Any]], tuple[str, ...] | None] | None = None,
    ) -> None:
        self._config = config
        self._resource = resource
        self._strategy = strategy
        self._field_map = field_map
        self._async_sql_resource = async_sql_resource
        self._df_params = df_params
        self._configured = configured
        self._build_configured_request = build_configured_request
        self._get_configured_select = get_configured_select
        self._get_configured_select_async = get_configured_select_async
        self._configured_fieldnames = configured_fieldnames

    def update_async_resource(self, resource: AsyncSqlDatabaseResource | None) -> None:
        self._async_sql_resource = resource

    def resolve(self, options: dict[str, Any]) -> ResolvedReturnType:
        if self._configured:
            return self._resolve_configured(options)
        return self._resolve_structured(options)

    async def resolve_async(self, options: dict[str, Any]) -> ResolvedReturnType:
        if self._configured:
            return await self._resolve_configured_async(options)
        return await self._resolve_structured_async(options)

    # ------------------------------------------------------------------
    # Structured mode
    # ------------------------------------------------------------------

    def _early_structured_decision(
        self, options: dict[str, Any], *, request: GatewayLoadRequest | None = None
    ) -> ResolvedReturnType | None:
        """Cheap checks that can decide the return type without probing the backend."""
        if request is not None:
            if request.partitioned is True:
                return "dask"
            if request.partitioned is False:
                return "pandas"
            limit = request.limit
            if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
                return "pandas"
        else:
            if options.get("partitioned") is True:
                return "dask"
            if options.get("partitioned") is False:
                return "pandas"
            limit = options.get("limit")
            if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
                return "pandas"
        if options.get("sql") is not None:
            return "pandas"
        return None

    def _build_structured_probe_context(
        self, options: dict[str, Any], request: GatewayLoadRequest | None
    ) -> StructuredLoadContext:
        return StructuredLoadContext(
            resource=self._resource,
            config=self._config,
            request=request,
            opts=options,
            loader_return_type="pandas",
            async_sql_resource=self._async_sql_resource,
        )

    def _resolve_structured(self, options: dict[str, Any], *, request: GatewayLoadRequest | None = None) -> ResolvedReturnType:
        decision = self._early_structured_decision(options, request=request)
        if decision is not None:
            return decision
        ctx = self._build_structured_probe_context(options, request)
        result = self._strategy.estimate_result_size(ctx)
        return self._decide(result)

    async def _resolve_structured_async(self, options: dict[str, Any], *, request: GatewayLoadRequest | None = None) -> ResolvedReturnType:
        decision = self._early_structured_decision(options, request=request)
        if decision is not None:
            return decision
        ctx = self._build_structured_probe_context(options, request)
        result = await self._strategy.estimate_result_size_async(ctx)
        return self._decide(result)

    # ------------------------------------------------------------------
    # Configured mode
    # ------------------------------------------------------------------

    @staticmethod
    def _early_configured_decision(req: Any) -> ResolvedReturnType | None:
        """Cheap checks that can decide the return type without probing the backend."""
        if req.control.get("partitioned") is True:
            return "dask"
        if req.control.get("partitioned") is False:
            return "pandas"
        limit = req.control.get("limit")
        if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
            return "pandas"
        return None

    def _build_configured_probe_context(self, req: Any) -> ConfiguredLoadContext:
        return ConfiguredLoadContext(
            resource=self._resource,
            config=self._config,
            control=GatewayLoadRequest.model_validate(req.control),
            combined_filters={},
            db_filters=req.db_filters,
            db_columns=req.db_columns,
            configured_fieldnames=req.configured_fieldnames,
            field_map=self._field_map,
            exclude=False,
            sticky_filters={},
            return_type="pandas",
            loader_return_type="pandas",
            loader_as_pandas=True,
            execution_mode="eager",
            post_processor=self._dummy_post_processor(),
            get_configured_select=self._get_configured_select,
            get_configured_select_async=self._get_configured_select_async,
            async_sql_resource=self._async_sql_resource,
            chunk_size=self._df_params.chunk_size,
        )

    def _resolve_configured(self, options: dict[str, Any]) -> ResolvedReturnType:
        req = self._build_configured_request(options)
        decision = self._early_configured_decision(req)
        if decision is not None:
            return decision
        ctx = self._build_configured_probe_context(req)
        result = self._strategy.estimate_result_size(ctx)
        return self._decide(result)

    async def _resolve_configured_async(self, options: dict[str, Any]) -> ResolvedReturnType:
        if self._resource is not None:
            return await asyncio.to_thread(self._resolve_configured, options)

        req = self._build_configured_request(options)
        decision = self._early_configured_decision(req)
        if decision is not None:
            return decision
        ctx = self._build_configured_probe_context(req)
        result = await self._strategy.estimate_result_size_async(ctx)
        return self._decide(result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decide(
        result: tuple[int | None, int | None] | None,
    ) -> ResolvedReturnType:
        if result is None:
            return "dask"
        estimated_rows, estimated_bytes = result
        if estimated_rows is not None and estimated_rows <= _AUTO_EAGER_MAX_ROWS:
            return "pandas"
        if estimated_bytes is not None and estimated_bytes <= _AUTO_EAGER_MAX_BYTES:
            return "pandas"
        return "dask"

    @staticmethod
    def _dummy_post_processor() -> Any:
        """Minimal no-op post-processor for probe-only contexts."""
        from boti_data.field_map import FieldMap

        from .post_process import PostProcessor
        from .requests import DataFrameOptions, DataFrameParams
        return PostProcessor(
            FieldMap({}),
            DataFrameParams(),
            DataFrameOptions(),
        )
