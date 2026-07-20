from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from boti_data.db import AsyncSqlDatabaseResource

from ._backend_strategies import ConfiguredLoadContext
from .build_context import ConfiguredSelectAccessors, GatewayBuildContext
from .frame_strategies import FrameResult
from .load_request import GatewayLoadRequest
from .requests import (
    ConfiguredRequest,
    ResolvedExecutionMode,
    ReturnType,
)

__all__ = [
    "ConfiguredLoadService",
]


@dataclass(frozen=True)
class _ConfiguredNormalization:
    control: GatewayLoadRequest
    combined_filters: dict[str, Any]
    db_filters: dict[str, Any]
    db_columns: list[str] | None
    configured_fieldnames: tuple[str, ...] | None


class ConfiguredLoadService:
    def __init__(
        self,
        ctx: GatewayBuildContext,
        *,
        table: str | None,
        sticky_filters: dict[str, Any],
        exclude: bool,
        select_accessors: ConfiguredSelectAccessors = ConfiguredSelectAccessors(),
    ) -> None:
        self._strategy = ctx.strategy
        self._table = table
        self._config = ctx.config
        self._resource = ctx.resource
        self._field_map = ctx.field_map
        self._sticky_filters = sticky_filters
        self._exclude = exclude
        self._df_params = ctx.df_params
        self._post_processor = ctx.post_processor
        self._async_sql_resource = ctx.async_sql_resource
        self._get_configured_select = select_accessors.get_configured_select
        self._get_configured_select_async = select_accessors.get_configured_select_async

    def update_async_resource(self, resource: AsyncSqlDatabaseResource | None) -> None:
        self._async_sql_resource = resource

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _configured_fieldnames(self, request: GatewayLoadRequest) -> tuple[str, ...] | None:
        if request.columns is not None:
            return tuple(request.columns)
        return self._df_params.fieldnames

    def _normalize_for_configured(self, options: dict[str, Any]) -> _ConfiguredNormalization:
        from .normalization import normalize_configured_filters

        normalized = normalize_configured_filters(
            options,
            sticky_filters=self._sticky_filters,
            exclude=self._exclude,
        )
        request = GatewayLoadRequest.model_validate(normalized.control)
        combined_filters = normalized.filters
        configured_fieldnames = self._configured_fieldnames(request)

        if self._field_map:
            db_filters = self._field_map.translate_filters_to_db(
                combined_filters, input_keys_are="semantic"
            )
            db_columns: list[str] | None = (
                self._field_map.select_db_columns(configured_fieldnames)
                if configured_fieldnames
                else None
            )
        else:
            db_filters = combined_filters
            db_columns = list(configured_fieldnames) if configured_fieldnames else None

        return _ConfiguredNormalization(
            control=request,
            combined_filters=combined_filters,
            db_filters=db_filters,
            db_columns=db_columns,
            configured_fieldnames=configured_fieldnames,
        )

    def _build_configured_request(self, options: dict[str, Any]) -> ConfiguredRequest:
        from boti_data.db import SqlDatabaseResource

        assert isinstance(self._resource, SqlDatabaseResource)
        norm = self._normalize_for_configured(options)
        assert self._get_configured_select is not None
        model, stmt = self._get_configured_select(norm.db_columns)
        return ConfiguredRequest(
            model=model,
            statement=stmt,
            db_filters=norm.db_filters,
            db_columns=norm.db_columns,
            control=norm.control.model_dump(),
            configured_fieldnames=norm.configured_fieldnames,
        )

    # ------------------------------------------------------------------
    # Sync configured load
    # ------------------------------------------------------------------

    def _build_configured_load_context(
        self,
        options: dict[str, Any],
        *,
        return_type: ReturnType,
        execution_mode: ResolvedExecutionMode,
        loader_return_type: Literal["pandas", "arrow", "dask"],
        loader_as_pandas: bool,
    ) -> ConfiguredLoadContext:
        norm = self._normalize_for_configured(options)
        return ConfiguredLoadContext(
            resource=self._resource,
            config=self._config,
            control=norm.control,
            combined_filters=norm.combined_filters,
            db_filters=norm.db_filters,
            db_columns=norm.db_columns,
            configured_fieldnames=norm.configured_fieldnames,
            field_map=self._field_map,
            exclude=self._exclude,
            sticky_filters=self._sticky_filters,
            return_type=return_type,
            loader_return_type=loader_return_type,
            loader_as_pandas=loader_as_pandas,
            execution_mode=execution_mode,
            post_processor=self._post_processor,
            get_configured_select=self._get_configured_select,
            get_configured_select_async=self._get_configured_select_async,
            async_sql_resource=self._async_sql_resource,
            chunk_size=self._df_params.chunk_size,
        )

    # Not a copy-pasted twin: both already build the shared ConfiguredLoadContext
    # via _build_configured_load_context(); the remaining difference
    # (self._strategy.load_configured_sync(ctx) vs await ...load_configured_async(ctx))
    # is the theoretical floor for this pattern.
    # spaghetti-ignore[sync-async-duplication]
    def load(
        self,
        options: dict[str, Any],
        *,
        return_type: ReturnType,
        execution_mode: ResolvedExecutionMode,
        loader_return_type: Literal["pandas", "arrow", "dask"],
        loader_as_pandas: bool,
    ) -> FrameResult:
        ctx = self._build_configured_load_context(
            options,
            return_type=return_type,
            execution_mode=execution_mode,
            loader_return_type=loader_return_type,
            loader_as_pandas=loader_as_pandas,
        )
        return self._strategy.load_configured_sync(ctx)

    # ------------------------------------------------------------------
    # Async configured load
    # ------------------------------------------------------------------

    async def aload(
        self,
        options: dict[str, Any],
        *,
        return_type: ReturnType,
        execution_mode: ResolvedExecutionMode,
        loader_return_type: Literal["pandas", "arrow", "dask"],
        loader_as_pandas: bool,
    ) -> FrameResult:
        ctx = self._build_configured_load_context(
            options,
            return_type=return_type,
            execution_mode=execution_mode,
            loader_return_type=loader_return_type,
            loader_as_pandas=loader_as_pandas,
        )
        return await self._strategy.load_configured_async(ctx)
