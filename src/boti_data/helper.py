from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from boti_dask import DaskSession, dask_session
from boti_data.gateway import DataGateway
from boti_data.gateway.requests import BackendConfig
from boti_data.joins import indexed_left_join, left_join_frames
from boti_data.schema import DataFrameLike


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

    def preview(self, *, n: int = 5, npartitions: int = 1, **options: Any) -> Any:
        return self._helper.preview(n=n, npartitions=npartitions, **self._bind_options(options))

    async def apreview(self, *, n: int = 5, npartitions: int = 1, **options: Any) -> Any:
        return await self._helper.apreview(n=n, npartitions=npartitions, **self._bind_options(options))

    def load_period(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return self._helper.load_period(dt_field, start, end, **self._bind_options(kwargs))

    async def aload_period(self, dt_field: str, start: str, end: str, **kwargs: Any) -> Any:
        return await self._helper.aload_period(dt_field, start, end, **self._bind_options(kwargs))

    def semi_join(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return self._helper.semi_join(join_series, on, **self._bind_options(kwargs))

    async def asemi_join(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return await self._helper.asemi_join(join_series, on, **self._bind_options(kwargs))


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

    def preview(self, *, n: int = 5, npartitions: int = 1, **options: Any) -> Any:
        return self.gateway.preview(n=n, npartitions=npartitions, **options)

    async def apreview(self, *, n: int = 5, npartitions: int = 1, **options: Any) -> Any:
        return await self.gateway.apreview(n=n, npartitions=npartitions, **options)

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

    def semi_join(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return self.gateway.semi_join(join_series, on, **kwargs)

    async def asemi_join(self, join_series: Any, on: str, **kwargs: Any) -> Any:
        return await self.gateway.asemi_join(join_series, on, **kwargs)

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


__all__ = ["DataHelper"]
