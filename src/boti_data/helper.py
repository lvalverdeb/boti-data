from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from boti_dask import DaskSession, dask_session

from boti_data.gateway import DataGateway
from boti_data.gateway.requests import BackendConfig
from boti_data.joins import indexed_left_join, left_join_frames
from boti_data.schema import DataFrameLike
from boti_data.watermark import (
    FileWatermarkStore,
    IncrementalResult,
    WatermarkStore,
    advance_watermark,
    build_incremental_filters,
)


class _EngineBoundHelper:
    """Thin engine-specific view over an existing :class:`DataHelper`."""

    def __init__(
        self,
        helper: DataHelper,
        *,
        return_type: str,
        execution_mode: str,
    ) -> None:
        self._helper = helper
        self._return_type = return_type
        self._execution_mode = execution_mode

    def _validate_options(self, options: Mapping[str, Any]) -> None:
        requested_return_type = options.get("return_type")
        if requested_return_type is not None and requested_return_type != self._return_type:
            raise ValueError(
                f"helper.{self._return_type} does not allow return_type={requested_return_type!r}; "
                f"use return_type={self._return_type!r} or call DataHelper.load(...) directly."
            )

        requested_execution_mode = options.get("execution_mode")
        if requested_execution_mode is not None and requested_execution_mode != self._execution_mode:
            raise ValueError(
                f"helper.{self._return_type} does not allow execution_mode={requested_execution_mode!r}; "
                f"use execution_mode={self._execution_mode!r} or call DataHelper.load(...) directly."
            )

        if options.get("as_pandas") and self._return_type != "pandas":
            raise ValueError(
                f"helper.{self._return_type} does not allow as_pandas=True; "
                "use helper.pandas or call DataHelper.load(...) directly."
            )

    def _bind_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_options(options)
        bound = dict(options)
        bound.setdefault("return_type", self._return_type)
        bound.setdefault("execution_mode", self._execution_mode)
        return bound

    def load(self, **options: Any) -> Any:
        return self._helper.load(**self._bind_options(options))

    async def aload(self, **options: Any) -> Any:
        return await self._helper.aload(**self._bind_options(options))

    def aload_sync(self, **options: Any) -> Any:
        return self._helper.aload_sync(**self._bind_options(options))

    def preview(self, *, n: int = 5, npartitions: int = 1, **options: Any) -> Any:
        return self._helper.preview(n=n, npartitions=npartitions, **self._bind_options(options))

    async def apreview(self, *, n: int = 5, npartitions: int = 1, **options: Any) -> Any:
        return await self._helper.apreview(n=n, npartitions=npartitions, **self._bind_options(options))

    def apreview_sync(self, *, n: int = 5, npartitions: int = 1, **options: Any) -> Any:
        return self._helper.apreview_sync(n=n, npartitions=npartitions, **self._bind_options(options))

    def load_period(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return self._helper.load_period(dt_field, start, end, **self._bind_options(kwargs))

    async def aload_period(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return await self._helper.aload_period(dt_field, start, end, **self._bind_options(kwargs))

    def aload_period_sync(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return self._helper.aload_period_sync(dt_field, start, end, **self._bind_options(kwargs))

    def semi_join(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return self._helper.semi_join(join_series, on, **self._bind_options(kwargs))

    async def asemi_join(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return await self._helper.asemi_join(join_series, on, **self._bind_options(kwargs))

    def asemi_join_sync(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return self._helper.asemi_join_sync(join_series, on, **self._bind_options(kwargs))

    def load_incremental(
        self,
        *,
        watermark_field: str,
        watermark_source: str | None = None,
        watermark_store: WatermarkStore | None = None,
        initial_value: Any = None,
        operator: str = "gt",
        commit_on_success: bool = True,
        **load_options: Any,
    ) -> IncrementalResult:
        return self._helper.load_incremental(
            watermark_field=watermark_field,
            watermark_source=watermark_source,
            watermark_store=watermark_store,
            initial_value=initial_value,
            operator=operator,
            commit_on_success=commit_on_success,
            **self._bind_options(load_options),
        )

    async def aload_incremental(
        self,
        *,
        watermark_field: str,
        watermark_source: str | None = None,
        watermark_store: WatermarkStore | None = None,
        initial_value: Any = None,
        operator: str = "gt",
        commit_on_success: bool = True,
        **load_options: Any,
    ) -> IncrementalResult:
        return await self._helper.aload_incremental(
            watermark_field=watermark_field,
            watermark_source=watermark_source,
            watermark_store=watermark_store,
            initial_value=initial_value,
            operator=operator,
            commit_on_success=commit_on_success,
            **self._bind_options(load_options),
        )

    def aload_incremental_sync(
        self,
        *,
        watermark_field: str,
        watermark_source: str | None = None,
        watermark_store: WatermarkStore | None = None,
        initial_value: Any = None,
        operator: str = "gt",
        commit_on_success: bool = True,
        **load_options: Any,
    ) -> IncrementalResult:
        return self._helper.aload_incremental_sync(
            watermark_field=watermark_field,
            watermark_source=watermark_source,
            watermark_store=watermark_store,
            initial_value=initial_value,
            operator=operator,
            commit_on_success=commit_on_success,
            **self._bind_options(load_options),
        )


class DataHelper:
    """Thin compatibility facade over :class:`DataGateway` and related helpers."""

    def __init__(
        self,
        config: DataGateway | BackendConfig | dict[str, Any] | None = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            if not overrides:
                raise TypeError("DataHelper requires a config object or legacy keyword config.")
            self.gateway = DataGateway.from_config(dict(overrides))
            return
        if isinstance(config, DataGateway):
            if overrides:
                raise TypeError("DataHelper does not accept overrides when wrapping an existing DataGateway.")
            self.gateway = config
        elif isinstance(config, Mapping):
            self.gateway = DataGateway.from_config(dict(config), **overrides)
        else:
            self.gateway = DataGateway(config, **overrides)

    @classmethod
    def from_legacy_config(cls, config: dict[str, Any], **overrides: Any) -> DataHelper:
        return cls(config, **overrides)

    @classmethod
    def from_gateway(cls, gateway: DataGateway) -> DataHelper:
        return cls(gateway)

    def __enter__(self) -> DataHelper:
        self.gateway.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.gateway.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> DataHelper:
        await self.gateway.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.gateway.__aexit__(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        self.gateway.close()

    async def aclose(self) -> None:
        await self.gateway.aclose()

    def load(self, **options: Any) -> Any:
        return self.gateway.load(**options)

    async def aload(self, **options: Any) -> Any:
        return await self.gateway.aload(**options)

    def aload_sync(self, **options: Any) -> Any:
        """Run :meth:`aload` from synchronous code.

        This helper is intentionally strict: when an event loop is already running
        (for example in notebooks), callers must use ``await helper.aload(...)``.
        """
        return self._run_coro_sync(self.aload, **options)

    def preview(self, *, n: int = 5, npartitions: int = 1, **options: Any) -> Any:
        return self.gateway.preview(n=n, npartitions=npartitions, **options)

    async def apreview(self, *, n: int = 5, npartitions: int = 1, **options: Any) -> Any:
        return await self.gateway.apreview(n=n, npartitions=npartitions, **options)

    def apreview_sync(self, *, n: int = 5, npartitions: int = 1, **options: Any) -> Any:
        return self._run_coro_sync(self.apreview, n=n, npartitions=npartitions, **options)

    @staticmethod
    def _run_coro_sync(coro_factory: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro_factory(*args, **kwargs))
        raise RuntimeError(
            "Cannot run async DataHelper operation from synchronous bridge while an event loop "
            "is running. Use `await helper.aload(...)` (or `await helper.<engine>.aload(...)`) "
            "in notebooks/async runtimes."
        )

    @property
    def dask(self) -> _EngineBoundHelper:
        return _EngineBoundHelper(self, return_type="dask", execution_mode="lazy")

    @property
    def pandas(self) -> _EngineBoundHelper:
        return _EngineBoundHelper(self, return_type="pandas", execution_mode="eager")

    @property
    def polars(self) -> _EngineBoundHelper:
        return _EngineBoundHelper(self, return_type="polars", execution_mode="eager")

    def load_period(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return self.gateway.load_period(dt_field, start, end, **kwargs)

    async def aload_period(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return await self.gateway.aload_period(dt_field, start, end, **kwargs)

    def aload_period_sync(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return self._run_coro_sync(self.aload_period, dt_field, start, end, **kwargs)

    def semi_join(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return self.gateway.semi_join(join_series, on, **kwargs)

    async def asemi_join(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return await self.gateway.asemi_join(join_series, on, **kwargs)

    def asemi_join_sync(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return self._run_coro_sync(self.asemi_join, join_series, on, **kwargs)

    def load_incremental(
        self,
        *,
        watermark_field: str,
        watermark_source: str | None = None,
        watermark_store: WatermarkStore | None = None,
        initial_value: Any = None,
        operator: str = "gt",
        commit_on_success: bool = True,
        **load_options: Any,
    ) -> IncrementalResult:
        resolved_source = watermark_source or getattr(self.gateway, "_table", None) or "default"
        resolved_store = watermark_store or FileWatermarkStore()
        previous = resolved_store.read(source=resolved_source)
        filters = {}
        if previous is not None:
            filters = build_incremental_filters(
                watermark_field=watermark_field,
                watermark_value=previous,
                operator=operator,
            )
        elif initial_value is not None:
            filters = build_incremental_filters(
                watermark_field=watermark_field,
                watermark_value=initial_value,
                operator=operator,
            )
        merged = {**load_options.pop("filters", {}), **filters}
        if merged:
            load_options["filters"] = merged
        frame = self.load(**load_options)
        records_loaded = len(frame) if hasattr(frame, "__len__") else 0
        current = advance_watermark(frame, watermark_field=watermark_field)
        committed = False
        if commit_on_success and current is not None:
            resolved_store.write(source=resolved_source, value=current)
            committed = True
        return IncrementalResult(
            frame=frame,
            watermark_field=watermark_field,
            previous_watermark=previous,
            current_watermark=current,
            records_loaded=records_loaded,
            watermark_committed=committed,
        )

    async def aload_incremental(
        self,
        *,
        watermark_field: str,
        watermark_source: str | None = None,
        watermark_store: WatermarkStore | None = None,
        initial_value: Any = None,
        operator: str = "gt",
        commit_on_success: bool = True,
        **load_options: Any,
    ) -> IncrementalResult:
        resolved_source = watermark_source or getattr(self.gateway, "_table", None) or "default"
        resolved_store = watermark_store or FileWatermarkStore()
        previous = resolved_store.read(source=resolved_source)
        filters = {}
        if previous is not None:
            filters = build_incremental_filters(
                watermark_field=watermark_field,
                watermark_value=previous,
                operator=operator,
            )
        elif initial_value is not None:
            filters = build_incremental_filters(
                watermark_field=watermark_field,
                watermark_value=initial_value,
                operator=operator,
            )
        merged = {**load_options.pop("filters", {}), **filters}
        if merged:
            load_options["filters"] = merged
        frame = await self.aload(**load_options)
        records_loaded = len(frame) if hasattr(frame, "__len__") else 0
        current = advance_watermark(frame, watermark_field=watermark_field)
        committed = False
        if commit_on_success and current is not None:
            resolved_store.write(source=resolved_source, value=current)
            committed = True
        return IncrementalResult(
            frame=frame,
            watermark_field=watermark_field,
            previous_watermark=previous,
            current_watermark=current,
            records_loaded=records_loaded,
            watermark_committed=committed,
        )

    def aload_incremental_sync(
        self,
        *,
        watermark_field: str,
        watermark_source: str | None = None,
        watermark_store: WatermarkStore | None = None,
        initial_value: Any = None,
        operator: str = "gt",
        commit_on_success: bool = True,
        **load_options: Any,
    ) -> IncrementalResult:
        return self._run_coro_sync(
            self.aload_incremental,
            watermark_field=watermark_field,
            watermark_source=watermark_source,
            watermark_store=watermark_store,
            initial_value=initial_value,
            operator=operator,
            commit_on_success=commit_on_success,
            **load_options,
        )

    @staticmethod
    def session(**kwargs: Any) -> DaskSession:
        return dask_session(**kwargs)

    @staticmethod
    def left_join(
        left: DataFrameLike,
        right: DataFrameLike,
        *,
        join_schema_map: Mapping[str, str],
        join_key: str | None = None,
        left_on: Sequence[str] | None = None,
        right_on: Sequence[str] | None = None,
        persist: bool = False,
        resilient: bool = False,
        reset_index: bool = True,
        diagnostics: bool = False,
        dry_run: bool = False,
        logger: Any | None = None,
        label: str | None = None,
    ) -> DataFrameLike:
        if join_key is not None:
            return indexed_left_join(
                left,
                right,
                join_key=join_key,
                join_schema_map=join_schema_map,
                persist=persist,
                resilient=resilient,
                reset_index=reset_index,
                diagnostics=diagnostics,
                dry_run=dry_run,
                logger=logger,
                label=label or "data_helper.indexed_left_join",
            )
        if left_on is None:
            raise ValueError("left_on is required when join_key is not provided.")
        return left_join_frames(
            left,
            right,
            left_on=left_on,
            right_on=right_on,
            join_schema_map=join_schema_map,
            diagnostics=diagnostics,
            dry_run=dry_run,
            logger=logger,
            label=label or "data_helper.left_join_frames",
        )


__all__ = ["DataHelper", "_EngineBoundHelper"]
