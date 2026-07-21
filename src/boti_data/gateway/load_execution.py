"""Structured/configured load execution pipeline for DataGateway.load()/aload()."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa

from boti_data.db import AsyncSqlDatabaseResource

from ._backend_strategies import StructuredLoadContext
from .build_context import GatewayBuildContext, LoadExecutorCallbacks, LoadValidationPolicy
from .configured_load import ConfiguredLoadService
from .frame_strategies import FrameResult
from .load_request import GatewayLoadRequest
from .normalization import split_control_and_filters
from .planning import LoadPlan, LoadPlanner
from .requests import ResolvedExecutionMode
from .sql_guard import validate_raw_sql_statement


def configured_call_kwargs(plan: LoadPlan) -> dict[str, Any]:
    """Shared kwargs for ConfiguredLoadService.load()/aload()."""
    return dict(
        return_type=plan.resolved_return_type,
        execution_mode=plan.resolved_execution_mode,
        loader_return_type=plan.loader_return_type,
        loader_as_pandas=plan.loader_as_pandas,
    )


def resolve_control_and_request(options: dict[str, Any]) -> GatewayLoadRequest:
    control, _ = split_control_and_filters(options)
    return GatewayLoadRequest.model_validate(control)


class LoadExecutor:
    """Runs a resolved load plan against the structured or configured backend.

    Split out of DataGateway: this is the largest single cluster of gateway
    internals, none of it referenced directly by tests, so it moves here
    wholesale rather than staying as thin wrappers.
    """

    def __init__(
        self,
        ctx: GatewayBuildContext,
        *,
        configured_loader: ConfiguredLoadService,
        validation_policy: LoadValidationPolicy,
        callbacks: LoadExecutorCallbacks,
    ) -> None:
        self._strategy = ctx.strategy
        self._configured = ctx.configured
        self._configured_loader = configured_loader
        self._resource = ctx.resource
        self._config = ctx.config
        self._post_processor = ctx.post_processor
        self._field_map = ctx.field_map
        self._raw_sql_policy = validation_policy.raw_sql_policy
        self._strict_filter_validation = validation_policy.strict_filter_validation
        self._allowed_filter_fields = validation_policy.allowed_filter_fields
        self._default_return_type = ctx.df_params.return_type
        self._default_execution_mode = ctx.df_params.execution_mode
        self._resolve_auto_return_type = callbacks.resolve_auto_return_type
        self._resolve_auto_return_type_async = callbacks.resolve_auto_return_type_async
        self._resolve_in_chunk_controls = callbacks.resolve_in_chunk_controls
        self._async_sql_resource = ctx.async_sql_resource

    def update_async_resource(self, resource: AsyncSqlDatabaseResource | None) -> None:
        self._async_sql_resource = resource

    def load_planner(self) -> LoadPlanner:
        return LoadPlanner(
            configured=self._configured,
            default_return_type=self._default_return_type,
            default_execution_mode=self._default_execution_mode,
            resolve_auto_return_type=self._resolve_auto_return_type,
            resolve_auto_return_type_async=self._resolve_auto_return_type_async,
        )

    def _prepare_structured_loader_options(self, options: dict[str, Any]) -> dict[str, Any]:
        if self._configured:
            return options
        return self._strategy.prepare_structured_options(options, self._field_map, self._configured)

    def _validate_raw_sql(self, options: dict[str, Any]) -> None:
        raw_sql = options.get("sql")
        if raw_sql is not None:
            allow_raw_sql = options.get("allow_raw_sql", False)
            if self._raw_sql_policy == "disabled":
                raise ValueError("Raw sql= execution is disabled by this DataGateway policy.")
            validate_raw_sql_statement(sql=raw_sql, allow_raw_sql=allow_raw_sql)

    def _validate_runtime_filters(self, loader_options: dict[str, Any]) -> None:
        if self._strict_filter_validation and self._configured:
            _ctrl, runtime_filters = split_control_and_filters(loader_options)
            for key in runtime_filters:
                field = key.split("__")[0]
                if field not in self._allowed_filter_fields:
                    raise ValueError(
                        f"Filter field '{field}' is not allowed. "
                        f"Allowed fields: {sorted(self._allowed_filter_fields)}"
                    )

    def build_structured_load_context(
        self,
        *,
        request: GatewayLoadRequest | None,
        opts: dict[str, Any],
        loader_return_type: Literal["pandas", "arrow", "dask"],
        resolved_execution_mode: ResolvedExecutionMode,
        timeout: float | None = None,
    ) -> StructuredLoadContext:
        return StructuredLoadContext(
            resource=self._resource,
            config=self._config,
            request=request,
            opts=opts,
            loader_return_type=loader_return_type,
            resolved_execution_mode=resolved_execution_mode,
            timeout=timeout,
            post_processor=self._post_processor,
            async_sql_resource=self._async_sql_resource,
        )

    # Not a copy-pasted twin: the configured-mode kwargs are already shared via
    # configured_call_kwargs(); the remaining difference (ctx-building +
    # asyncio.wait_for timeout wrapping in the async path) is a real, timeout
    # feature that only the async path supports.
    # spaghetti-ignore[sync-async-duplication]: see above
    def perform_load_sync(
        self,
        opts: dict[str, Any],
        *,
        plan: LoadPlan,
        request: GatewayLoadRequest | None = None,
    ) -> pd.DataFrame | dd.DataFrame | pa.Table:
        if self._configured:
            return self._configured_loader.load(opts, **configured_call_kwargs(plan))
        ctx = self.build_structured_load_context(
            request=request,
            opts=opts,
            loader_return_type=plan.loader_return_type,
            resolved_execution_mode=plan.resolved_execution_mode,
        )
        return self._strategy.load_structured_sync(ctx)

    async def perform_load_async(
        self,
        opts: dict[str, Any],
        *,
        plan: LoadPlan,
        request: GatewayLoadRequest | None = None,
        timeout: float | None = None,
    ) -> pd.DataFrame | dd.DataFrame | pa.Table:
        if self._configured:
            coro = self._configured_loader.aload(opts, **configured_call_kwargs(plan))
        else:
            ctx = self.build_structured_load_context(
                request=request,
                opts=opts,
                loader_return_type=plan.loader_return_type,
                resolved_execution_mode=plan.resolved_execution_mode,
                timeout=timeout,
            )
            coro = self._strategy.load_structured_async(ctx)

        if timeout is not None:
            return await asyncio.wait_for(coro, timeout)
        return await coro

    def prepare_load_options(self, options: dict[str, Any], plan: LoadPlan) -> dict[str, Any]:
        self._validate_raw_sql(options)
        controls = plan.controls
        if controls.diagnostics:
            self._post_processor.log_load_start(plan)
        loader_options = self._prepare_structured_loader_options(
            {**options, "as_pandas": plan.loader_as_pandas}
        )
        if controls.diagnostics:
            loader_options["diagnostics"] = True
        return loader_options

    def finalize_load(self, df: FrameResult, plan: LoadPlan) -> FrameResult:
        return self._post_processor.finalize_load_result(df, plan)

    def resolve_in_chunk_controls_for_plan(
        self, loader_options: dict[str, Any], plan: LoadPlan
    ) -> tuple[int, int | None]:
        self._validate_runtime_filters(loader_options)
        controls = plan.controls
        return self._resolve_in_chunk_controls(
            loader_options,
            strategy=controls.in_chunk_strategy,
            execution_mode=plan.resolved_execution_mode,
            in_chunk_size_raw=controls.in_chunk_size_raw,
            in_chunk_concurrency_raw=controls.in_chunk_concurrency_raw,
        )
