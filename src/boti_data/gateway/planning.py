from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from .frame_strategies import FrameStrategy, get_frame_strategy
from .load_request import GatewayLoadRequest
from .requests import (
    ExecutionMode,
    ResolvedExecutionMode,
    ResolvedReturnType,
    ReturnType,
)

LoaderReturnType = Literal["pandas", "arrow", "dask"]
SyncAutoResolver = Callable[[dict[str, Any]], ResolvedReturnType]
AsyncAutoResolver = Callable[[dict[str, Any]], Awaitable[ResolvedReturnType]]


@dataclass
class InChunkPolicy:
    """Controls automatic fan-out for large SQL ``IN`` filters."""

    eager_auto_min_values: int
    eager_auto_concurrency: int | None = None


@dataclass(frozen=True)
class LoadControls:
    persist: bool
    resilient: bool
    diagnostics: bool
    as_pandas: bool
    in_chunk_strategy: str
    in_chunk_size_raw: Any
    in_chunk_concurrency_raw: Any
    dry_run: bool
    timeout: float | None


@dataclass(frozen=True)
class LoadPlan:
    controls: LoadControls
    started: float
    requested_return_type: ReturnType
    requested_execution_mode: ExecutionMode
    resolved_return_type: ResolvedReturnType
    resolved_execution_mode: ResolvedExecutionMode
    loader_return_type: LoaderReturnType
    loader_as_pandas: bool
    strategy: FrameStrategy


class LoadPlanner:
    """Resolves public load options into an executable backend plan."""

    def __init__(
        self,
        *,
        configured: bool,
        default_return_type: ReturnType,
        default_execution_mode: ExecutionMode,
        resolve_auto_return_type: SyncAutoResolver,
        resolve_auto_return_type_async: AsyncAutoResolver,
    ) -> None:
        self._configured = configured
        self._default_return_type = default_return_type
        self._default_execution_mode = default_execution_mode
        self._resolve_auto_return_type = resolve_auto_return_type
        self._resolve_auto_return_type_async = resolve_auto_return_type_async

    def _prepare_plan_inputs(
        self, options: dict[str, Any], *, request: GatewayLoadRequest | None = None
    ) -> tuple[LoadControls, ReturnType, ExecutionMode]:
        controls = self._resolve_load_controls(options, request=request)
        requested_return_type = self._resolve_requested_return_type(
            as_pandas=controls.as_pandas,
            options=options,
            request=request,
        )
        requested_execution_mode = self._resolve_requested_execution_mode(options, request=request)
        return controls, requested_return_type, requested_execution_mode

    # Not a copy-pasted twin: shared prep (_prepare_plan_inputs) and shared
    # finish (_finish_plan) are already extracted; the remaining difference is
    # the irreducible self._resolve_execution_plan(...) vs
    # await ..._async(...) call.
    # spaghetti-ignore[sync-async-duplication]
    def plan(
        self, options: dict[str, Any], *, request: GatewayLoadRequest | None = None
    ) -> LoadPlan:
        controls, requested_return_type, requested_execution_mode = self._prepare_plan_inputs(
            options, request=request
        )
        resolved = self._resolve_execution_plan(
            requested_return_type=requested_return_type,
            requested_execution_mode=requested_execution_mode,
            options=options,
        )
        return self._finish_plan(
            controls=controls,
            requested_return_type=requested_return_type,
            requested_execution_mode=requested_execution_mode,
            resolved=resolved,
        )

    async def aplan(
        self, options: dict[str, Any], *, request: GatewayLoadRequest | None = None
    ) -> LoadPlan:
        controls, requested_return_type, requested_execution_mode = self._prepare_plan_inputs(
            options, request=request
        )
        resolved = await self._resolve_execution_plan_async(
            requested_return_type=requested_return_type,
            requested_execution_mode=requested_execution_mode,
            options=options,
        )
        return self._finish_plan(
            controls=controls,
            requested_return_type=requested_return_type,
            requested_execution_mode=requested_execution_mode,
            resolved=resolved,
        )

    def _finish_plan(
        self,
        *,
        controls: LoadControls,
        requested_return_type: ReturnType,
        requested_execution_mode: ExecutionMode,
        resolved: tuple[ResolvedReturnType, ResolvedExecutionMode],
    ) -> LoadPlan:
        resolved_return_type, resolved_execution_mode = resolved
        return self._build_plan(
            controls=controls,
            requested_return_type=requested_return_type,
            requested_execution_mode=requested_execution_mode,
            resolved_return_type=resolved_return_type,
            resolved_execution_mode=resolved_execution_mode,
        )

    @staticmethod
    def _resolve_load_controls(
        options: dict[str, Any],
        *,
        request: GatewayLoadRequest | None = None,
    ) -> LoadControls:
        if request is not None:
            return LoadControls(
                persist=request.persist,
                resilient=request.resilient,
                diagnostics=request.diagnostics,
                as_pandas=request.as_pandas,
                in_chunk_strategy=request.in_chunk_strategy,
                in_chunk_size_raw=request.in_chunk_size,
                in_chunk_concurrency_raw=request.in_chunk_concurrency,
                dry_run=request.dry_run,
                timeout=request.timeout,
            )
        return LoadControls(
            persist=bool(options.pop("persist", False)),
            resilient=bool(options.pop("resilient", False)),
            diagnostics=bool(options.pop("diagnostics", False)),
            as_pandas=bool(options.get("as_pandas", False)),
            in_chunk_strategy=options.pop("in_chunk_strategy", "auto"),
            in_chunk_size_raw=options.pop("in_chunk_size", None),
            in_chunk_concurrency_raw=options.pop("in_chunk_concurrency", None),
            dry_run=bool(options.pop("dry_run", False)),
            timeout=options.pop("timeout", None),
        )

    @staticmethod
    def _explicit_requested_return_type(
        *,
        options: dict[str, Any],
        request: GatewayLoadRequest | None,
    ) -> ReturnType | None:
        """Returns an explicitly-requested return type, or None if unspecified."""
        if request is not None and request.return_type is not None:
            return request.return_type
        if "return_type" in options:
            return options.pop("return_type")
        return None

    def _resolve_requested_return_type(
        self,
        *,
        as_pandas: bool,
        options: dict[str, Any],
        request: GatewayLoadRequest | None = None,
    ) -> ReturnType:
        explicit = self._explicit_requested_return_type(options=options, request=request)
        if explicit is not None:
            return explicit
        if as_pandas:
            return "pandas"
        return self._default_return_type if self._configured else "dask"

    @staticmethod
    def _explicit_requested_execution_mode(
        options: dict[str, Any],
        *,
        request: GatewayLoadRequest | None,
    ) -> ExecutionMode | None:
        """Returns an explicitly-requested execution mode, or None if unspecified."""
        if request is not None and request.execution_mode is not None:
            return request.execution_mode
        if "execution_mode" in options:
            return options.pop("execution_mode")
        return None

    @staticmethod
    def _execution_mode_from_partitioned(partitioned: bool | None) -> ExecutionMode | None:
        """Returns the execution mode implied by an explicit `partitioned` flag, else None."""
        if partitioned is True:
            return "lazy"
        if partitioned is False:
            return "eager"
        return None

    def _resolve_requested_execution_mode(
        self,
        options: dict[str, Any],
        *,
        request: GatewayLoadRequest | None = None,
    ) -> ExecutionMode:
        explicit = self._explicit_requested_execution_mode(options, request=request)
        if explicit is not None:
            return explicit
        partitioned = request.partitioned if request is not None else options.get("partitioned")
        from_partitioned = self._execution_mode_from_partitioned(partitioned)
        if from_partitioned is not None:
            return from_partitioned
        return self._default_execution_mode if self._configured else "auto"

    def _finish_execution_plan(
        self,
        *,
        resolved_return_type: ResolvedReturnType,
        requested_execution_mode: ExecutionMode,
    ) -> tuple[ResolvedReturnType, ResolvedExecutionMode]:
        return resolved_return_type, self._resolve_execution_mode(
            requested_execution_mode=requested_execution_mode,
            resolved_return_type=resolved_return_type,
        )

    # Not a copy-pasted twin: the shared tail is already extracted into
    # _finish_execution_plan(); the remaining difference is the irreducible
    # self._resolve_return_type(...) vs await ..._async(...) call.
    # spaghetti-ignore[sync-async-duplication]
    def _resolve_execution_plan(
        self,
        *,
        requested_return_type: ReturnType,
        requested_execution_mode: ExecutionMode,
        options: dict[str, Any],
    ) -> tuple[ResolvedReturnType, ResolvedExecutionMode]:
        resolved_return_type = self._resolve_return_type(
            requested_return_type=requested_return_type,
            requested_execution_mode=requested_execution_mode,
            options=options,
        )
        return self._finish_execution_plan(
            resolved_return_type=resolved_return_type,
            requested_execution_mode=requested_execution_mode,
        )

    async def _resolve_execution_plan_async(
        self,
        *,
        requested_return_type: ReturnType,
        requested_execution_mode: ExecutionMode,
        options: dict[str, Any],
    ) -> tuple[ResolvedReturnType, ResolvedExecutionMode]:
        resolved_return_type = await self._resolve_return_type_async(
            requested_return_type=requested_return_type,
            requested_execution_mode=requested_execution_mode,
            options=options,
        )
        return self._finish_execution_plan(
            resolved_return_type=resolved_return_type,
            requested_execution_mode=requested_execution_mode,
        )

    @staticmethod
    def _requested_or_fixed_return_type(
        *,
        requested_return_type: ReturnType,
        requested_execution_mode: ExecutionMode,
    ) -> ResolvedReturnType | None:
        """Returns a decision that doesn't require consulting the auto-resolver, else None."""
        if requested_return_type != "auto":
            return requested_return_type
        if requested_execution_mode != "auto":
            return "dask" if requested_execution_mode == "lazy" else "pandas"
        return None

    # Not a copy-pasted twin: shared decision logic is already extracted into
    # _requested_or_fixed_return_type(); the remaining difference is the
    # irreducible self._resolve_auto_return_type(options) vs
    # await ..._async(options) call.
    # spaghetti-ignore[sync-async-duplication]
    def _resolve_return_type(
        self,
        *,
        requested_return_type: ReturnType,
        requested_execution_mode: ExecutionMode,
        options: dict[str, Any],
    ) -> ResolvedReturnType:
        decision = self._requested_or_fixed_return_type(
            requested_return_type=requested_return_type,
            requested_execution_mode=requested_execution_mode,
        )
        if decision is not None:
            return decision
        return self._resolve_auto_return_type(options)

    async def _resolve_return_type_async(
        self,
        *,
        requested_return_type: ReturnType,
        requested_execution_mode: ExecutionMode,
        options: dict[str, Any],
    ) -> ResolvedReturnType:
        decision = self._requested_or_fixed_return_type(
            requested_return_type=requested_return_type,
            requested_execution_mode=requested_execution_mode,
        )
        if decision is not None:
            return decision
        return await self._resolve_auto_return_type_async(options)

    @staticmethod
    def _resolve_execution_mode(
        *,
        requested_execution_mode: ExecutionMode,
        resolved_return_type: ResolvedReturnType,
    ) -> ResolvedExecutionMode:
        if requested_execution_mode != "auto":
            return requested_execution_mode
        return "lazy" if resolved_return_type == "dask" else "eager"

    def _build_plan(
        self,
        *,
        controls: LoadControls,
        requested_return_type: ReturnType,
        requested_execution_mode: ExecutionMode,
        resolved_return_type: ResolvedReturnType,
        resolved_execution_mode: ResolvedExecutionMode,
    ) -> LoadPlan:
        self._validate_dry_run(controls.dry_run, resolved_return_type)
        loader_return_type = self._loader_return_type(
            resolved_return_type,
            execution_mode=resolved_execution_mode,
        )
        return LoadPlan(
            controls=controls,
            started=perf_counter(),
            requested_return_type=requested_return_type,
            requested_execution_mode=requested_execution_mode,
            resolved_return_type=resolved_return_type,
            resolved_execution_mode=resolved_execution_mode,
            loader_return_type=loader_return_type,
            loader_as_pandas=loader_return_type == "pandas",
            strategy=get_frame_strategy(resolved_return_type),
        )

    @staticmethod
    def _loader_return_type(
        return_type: ResolvedReturnType,
        *,
        execution_mode: ResolvedExecutionMode,
    ) -> LoaderReturnType:
        if execution_mode == "lazy":
            return "dask"
        return get_frame_strategy(return_type).loader_return_type

    @staticmethod
    def _validate_dry_run(dry_run: bool, resolved_return_type: str) -> None:
        if dry_run and resolved_return_type in ("pandas", "polars"):
            raise ValueError(
                "dry_run=True is only supported for lazy return types (dask). "
                "Use return_type='dask' for dry-run mode."
            )
