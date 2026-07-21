from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from boti_data.db import AsyncSqlDatabaseResource

from ._backend_strategies import ConfiguredLoadContext, StructuredLoadContext
from .build_context import ConfiguredSelectAccessors, GatewayBuildContext
from .load_request import GatewayLoadRequest
from .requests import ConfiguredRequest, ResolvedReturnType

_AUTO_EAGER_MAX_ROWS = 10_000
_AUTO_EAGER_MAX_BYTES = 32 * 1024 * 1024


class AutoReturnTypeResolver:
    """Resolves ``return_type='auto'`` by probing row counts or file sizes."""

    def __init__(
        self,
        ctx: GatewayBuildContext,
        *,
        build_configured_request: Callable[[dict[str, Any]], ConfiguredRequest],
        select_accessors: ConfiguredSelectAccessors = ConfiguredSelectAccessors(),
        configured_fieldnames: Callable[[dict[str, Any]], tuple[str, ...] | None] | None = None,
    ) -> None:
        self._config = ctx.config
        self._resource = ctx.resource
        self._strategy = ctx.strategy
        self._field_map = ctx.field_map
        self._async_sql_resource = ctx.async_sql_resource
        self._df_params = ctx.df_params
        self._configured = ctx.configured
        self._build_configured_request = build_configured_request
        self._get_configured_select = select_accessors.get_configured_select
        self._get_configured_select_async = select_accessors.get_configured_select_async
        self._configured_fieldnames = configured_fieldnames

    def update_async_resource(self, resource: AsyncSqlDatabaseResource | None) -> None:
        self._async_sql_resource = resource

    # Not a copy-pasted twin: already the thinnest possible sync/async
    # dispatch to the (also already-deduplicated) _resolve_configured(_async)/
    # _resolve_structured(_async) pairs below.
    # spaghetti-ignore[sync-async-duplication]: see above
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

    @staticmethod
    def _extract_partitioned_and_limit(
        options: dict[str, Any], request: GatewayLoadRequest | None
    ) -> tuple[bool | None, Any]:
        if request is not None:
            return request.partitioned, request.limit
        return options.get("partitioned"), options.get("limit")

    def _early_structured_decision(
        self, options: dict[str, Any], *, request: GatewayLoadRequest | None = None
    ) -> ResolvedReturnType | None:
        """Cheap checks that can decide the return type without probing the backend."""
        partitioned, limit = self._extract_partitioned_and_limit(options, request)
        if partitioned is True:
            return "dask"
        if (
            partitioned is False
            or (isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS)
            or options.get("sql") is not None
        ):
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

    # Not a copy-pasted twin: shared _early_structured_decision()/
    # _build_structured_probe_context() are already extracted; the remaining
    # difference is the irreducible self._strategy.estimate_result_size(ctx)
    # vs await ..._async(ctx) call.
    # spaghetti-ignore[sync-async-duplication]: see above
    def _resolve_structured(
        self, options: dict[str, Any], *, request: GatewayLoadRequest | None = None
    ) -> ResolvedReturnType:
        decision = self._early_structured_decision(options, request=request)
        if decision is not None:
            return decision
        ctx = self._build_structured_probe_context(options, request)
        result = self._strategy.estimate_result_size(ctx)
        return self._decide(result)

    async def _resolve_structured_async(
        self, options: dict[str, Any], *, request: GatewayLoadRequest | None = None
    ) -> ResolvedReturnType:
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
        partitioned = req.control.get("partitioned")
        if partitioned is True:
            return "dask"
        limit = req.control.get("limit")
        if partitioned is False or (isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS):
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

    def _prepare_configured_probe(
        self, options: dict[str, Any]
    ) -> tuple[ResolvedReturnType | None, ConfiguredLoadContext | None]:
        """Shared by _resolve_configured/_resolve_configured_async: builds the
        configured request and either an early decision (no probe needed) or
        the probe context to estimate against."""
        req = self._build_configured_request(options)
        decision = self._early_configured_decision(req)
        if decision is not None:
            return decision, None
        return None, self._build_configured_probe_context(req)

    # Not a copy-pasted twin: shared req/decision/context building is already
    # extracted into _prepare_configured_probe(); async also has a genuine
    # extra asyncio.to_thread(self._resolve_configured, ...) short-circuit
    # when ctx.resource is set, which sync has no equivalent for.
    # spaghetti-ignore[sync-async-duplication]: see above
    def _resolve_configured(self, options: dict[str, Any]) -> ResolvedReturnType:
        decision, ctx = self._prepare_configured_probe(options)
        if decision is not None:
            return decision
        result = self._strategy.estimate_result_size(ctx)
        return self._decide(result)

    async def _resolve_configured_async(self, options: dict[str, Any]) -> ResolvedReturnType:
        if self._resource is not None:
            return await asyncio.to_thread(self._resolve_configured, options)

        decision, ctx = self._prepare_configured_probe(options)
        if decision is not None:
            return decision
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
        within_row_threshold = estimated_rows is not None and estimated_rows <= _AUTO_EAGER_MAX_ROWS
        within_byte_threshold = (
            estimated_bytes is not None and estimated_bytes <= _AUTO_EAGER_MAX_BYTES
        )
        return "pandas" if within_row_threshold or within_byte_threshold else "dask"

    @staticmethod
    def _dummy_post_processor() -> Any:
        """Minimal no-op post-processor for probe-only contexts."""
        from boti_data.field_map import FieldMap

        from .post_process import PostProcessor, ResultShapingConfig
        from .requests import DataFrameOptions, DataFrameParams

        return PostProcessor(
            ResultShapingConfig(FieldMap({}), DataFrameParams(), DataFrameOptions()),
        )
