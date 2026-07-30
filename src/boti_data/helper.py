from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from boti.core.lifecycle import LifecycleCore
from boti.core.lifecycle_pickle import PicklableLifecycleCoreMixin
from boti_dask import DaskSession, dask_session

from boti_data.gateway import DataGateway
from boti_data.gateway.incremental import IncrementalLoadService
from boti_data.gateway.requests import BackendConfig, TableDescription
from boti_data.joins import indexed_left_join, left_join_frames
from boti_data.schema import DataFrameLike
from boti_data.watermark.incremental import IncrementalResult


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
        if (
            requested_execution_mode is not None
            and requested_execution_mode != self._execution_mode
        ):
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

    def load_period(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return self._helper.load_period(dt_field, start, end, **self._bind_options(kwargs))

    def aload_period_sync(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return self._helper.aload_period_sync(dt_field, start, end, **self._bind_options(kwargs))

    async def aload_period(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return await self._helper.aload_period(dt_field, start, end, **self._bind_options(kwargs))

    def semi_join(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return self._helper.semi_join(join_series, on, **self._bind_options(kwargs))

    async def asemi_join(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return await self._helper.asemi_join(join_series, on, **self._bind_options(kwargs))

    def asemi_join_sync(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return self._helper.asemi_join_sync(join_series, on, **self._bind_options(kwargs))

    # Not a copy-pasted twin: pure delegate to self._helper.load_incremental()/
    # aload_incremental() via the shared _bind_options(), identical in shape to
    # this class's other unflagged proxy pairs (load/aload, semi_join/asemi_join).
    # spaghetti-ignore[sync-async-duplication]: see above
    def load_incremental(
        self,
        *,
        watermark_field: str,
        watermark_source: str,
        watermark_store: Any,
        operator: str = "gt",
        **options: Any,
    ) -> IncrementalResult:
        return self._helper.load_incremental(
            watermark_field=watermark_field,
            watermark_source=watermark_source,
            watermark_store=watermark_store,
            operator=operator,
            **self._bind_options(options),
        )

    async def aload_incremental(
        self,
        *,
        watermark_field: str,
        watermark_source: str,
        watermark_store: Any,
        operator: str = "gt",
        **options: Any,
    ) -> IncrementalResult:
        return await self._helper.aload_incremental(
            watermark_field=watermark_field,
            watermark_source=watermark_source,
            watermark_store=watermark_store,
            operator=operator,
            **self._bind_options(options),
        )


def _session(**kwargs: Any) -> DaskSession:
    return dask_session(**kwargs)


def _left_join(
    left: DataFrameLike,
    right: DataFrameLike,
    *,
    join_schema_map: Mapping[str, str],
    join_key: str | None = None,
    left_on: Sequence[str] | None = None,
    right_on: Sequence[str] | None = None,
    persist: bool = False,
    reset_index: bool = True,
    diagnostics: bool = False,
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
            reset_index=reset_index,
            diagnostics=diagnostics,
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
        logger=logger,
        label=label or "data_helper.left_join_frames",
    )


class DataHelper(PicklableLifecycleCoreMixin, LifecycleCore):
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
        elif isinstance(config, DataGateway):
            if overrides:
                raise TypeError(
                    "DataHelper does not accept overrides when wrapping an existing DataGateway."
                )
            self.gateway = config
        elif isinstance(config, Mapping):
            self.gateway = DataGateway.from_config(dict(config), **overrides)
        else:
            self.gateway = DataGateway(config, **overrides)

        self._incremental = IncrementalLoadService(self.gateway)
        super().__init__()

    @classmethod
    def from_legacy_config(cls, config: dict[str, Any], **overrides: Any) -> DataHelper:
        return cls(config, **overrides)

    @classmethod
    def from_gateway(cls, gateway: DataGateway) -> DataHelper:
        return cls(gateway)

    def __enter__(self) -> DataHelper:
        super().__enter__()
        self.gateway.__enter__()
        return self

    async def __aenter__(self) -> DataHelper:
        await super().__aenter__()
        await self.gateway.__aenter__()
        return self

    def _cleanup(self) -> None:
        self.gateway.close()

    async def _acleanup(self) -> None:
        await self.gateway.aclose()

    def load(self, **options: Any) -> Any:
        return self.gateway.load(**options)

    async def aload(self, **options: Any) -> Any:
        return await self.gateway.aload(**options)

    def aload_sync(self, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aload(**kwargs))
        raise RuntimeError(
            "Use `await helper.aload(...)` instead of aload_sync() when an event loop is running."
        )

    def aload_period_sync(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aload_period(dt_field, start, end, **kwargs))
        raise RuntimeError(
            "Use `await helper.aload_period(...)` instead of aload_period_sync() when an event loop is running."
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

    @staticmethod
    def _build_preview_options(
        statement: Any, model: Any, n: int, options: dict[str, Any]
    ) -> dict[str, Any]:
        options["statement"] = statement
        options["model"] = model
        options["as_pandas"] = True
        options["limit"] = n
        return options

    def preview(self, statement: Any, model: Any, n: int = 5, **options: Any) -> Any:
        return self.gateway.load(**self._build_preview_options(statement, model, n, options))

    async def apreview(self, statement: Any, model: Any, n: int = 5, **options: Any) -> Any:
        return await self.gateway.aload(**self._build_preview_options(statement, model, n, options))

    # Not a copy-pasted twin: pure delegate to self.gateway.describe()/
    # adescribe() with identical kwargs, matching this class's other
    # unflagged proxy pairs (load/aload, preview/apreview).
    # spaghetti-ignore[sync-async-duplication]: see above
    def describe(
        self, table: str | None = None, *, row_count_limit: int = 10_000
    ) -> TableDescription:
        return self.gateway.describe(table, row_count_limit=row_count_limit)

    async def adescribe(
        self, table: str | None = None, *, row_count_limit: int = 10_000
    ) -> TableDescription:
        return await self.gateway.adescribe(table, row_count_limit=row_count_limit)

    def load_period(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return self.gateway.load_period(dt_field, start, end, **kwargs)

    async def aload_period(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return await self.gateway.aload_period(dt_field, start, end, **kwargs)

    def semi_join(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return self.gateway.semi_join(join_series, on, **kwargs)

    async def asemi_join(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return await self.gateway.asemi_join(join_series, on, **kwargs)

    def asemi_join_sync(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.asemi_join(join_series, on, **kwargs))
        raise RuntimeError(
            "Use `await helper.asemi_join(...)` instead of asemi_join_sync() when an event loop is running."
        )

    # Not a copy-pasted twin: pure delegate to self._incremental.load()/aload()
    # with identical kwargs, matching this class's other unflagged proxy pairs
    # (load/aload, preview/apreview, load_period/aload_period).
    # spaghetti-ignore[sync-async-duplication]: see above
    def load_incremental(
        self,
        *,
        watermark_field: str,
        watermark_source: str,
        watermark_store: Any,
        operator: str = "gt",
        initial_value: Any = None,
        commit_on_success: bool = True,
        **options: Any,
    ) -> IncrementalResult:
        return self._incremental.load(
            watermark_field=watermark_field,
            watermark_source=watermark_source,
            watermark_store=watermark_store,
            operator=operator,
            initial_value=initial_value,
            commit_on_success=commit_on_success,
            **options,
        )

    async def aload_incremental(
        self,
        *,
        watermark_field: str,
        watermark_source: str,
        watermark_store: Any,
        operator: str = "gt",
        initial_value: Any = None,
        commit_on_success: bool = True,
        **options: Any,
    ) -> IncrementalResult:
        return await self._incremental.aload(
            watermark_field=watermark_field,
            watermark_source=watermark_source,
            watermark_store=watermark_store,
            operator=operator,
            initial_value=initial_value,
            commit_on_success=commit_on_success,
            **options,
        )

    session = staticmethod(_session)
    left_join = staticmethod(_left_join)


__all__ = ["DataHelper"]
