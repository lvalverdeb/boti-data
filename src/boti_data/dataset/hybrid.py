from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
from boti_dask import safe_persist

from boti_data.datacube import DatacubeConfig
from boti_data.db import SqlDatabaseConfig
from boti_data.helper import DataHelper
from boti_data.parquet.resource import ParquetDataConfig

HybridSource = Literal["auto", "historical", "live"]
ResolvedReturnType = Literal["pandas", "arrow", "dask", "polars"]
BackendConfig = SqlDatabaseConfig | ParquetDataConfig | DatacubeConfig
FrameResult = pd.DataFrame | dd.DataFrame | pa.Table | pl.DataFrame


def _to_pandas(frame: FrameResult) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    if isinstance(frame, dd.DataFrame):
        return frame.compute()
    if isinstance(frame, pa.Table):
        return frame.to_pandas()
    if isinstance(frame, pl.DataFrame):
        return frame.to_pandas()
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _to_arrow(frame: FrameResult) -> pa.Table:
    if isinstance(frame, pa.Table):
        return frame
    if isinstance(frame, dd.DataFrame):
        frame = frame.compute()
    if isinstance(frame, pd.DataFrame):
        return pa.Table.from_pandas(frame, preserve_index=False)
    if isinstance(frame, pl.DataFrame):
        return frame.to_arrow()
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _to_polars(frame: FrameResult) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame
    return pl.from_arrow(_to_arrow(frame))


def _to_dask(frame: FrameResult) -> dd.DataFrame:
    if isinstance(frame, dd.DataFrame):
        return frame
    return dd.from_pandas(_to_pandas(frame), npartitions=1)


@dataclass(slots=True)
class _LoadPlan:
    source: HybridSource
    historical_start: str | None = None
    historical_end: str | None = None
    live_start: str | None = None
    live_end: str | None = None


class _HybridEngineBoundDataset:
    """Engine-specific view over an existing :class:`HybridDataset`."""

    def __init__(
        self,
        dataset: HybridDataset,
        *,
        return_type: ResolvedReturnType,
        execution_mode: Literal["lazy", "eager"],
    ) -> None:
        self._dataset = dataset
        self._return_type = return_type
        self._execution_mode = execution_mode

    def _bind_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        bound = dict(options)
        bound.setdefault("return_type", self._return_type)
        bound.setdefault("execution_mode", self._execution_mode)
        return bound

    def load(
        self,
        *,
        start: str,
        end: str,
        source: HybridSource = "auto",
        **options: Any,
    ) -> FrameResult:
        return self._dataset.load(
            start=start,
            end=end,
            source=source,
            **self._bind_options(options),
        )

    async def aload(
        self,
        *,
        start: str,
        end: str,
        source: HybridSource = "auto",
        **options: Any,
    ) -> FrameResult:
        return await self._dataset.aload(
            start=start,
            end=end,
            source=source,
            **self._bind_options(options),
        )


class HybridDataset:
    """Compose historical + live datasets behind one date-aware load API."""

    def __init__(
        self,
        historical: DataHelper | BackendConfig | Mapping[str, Any],
        live: DataHelper | BackendConfig | Mapping[str, Any],
        *,
        date_field: str,
        split_date: str | dt.date,
    ) -> None:
        self.historical = self._coerce_helper(historical)
        self.live = self._coerce_helper(live)
        self.date_field = date_field
        self.split_date = self._normalize_date(split_date)

    @staticmethod
    def _coerce_helper(
        helper_or_config: DataHelper | BackendConfig | Mapping[str, Any],
    ) -> DataHelper:
        if isinstance(helper_or_config, DataHelper):
            return helper_or_config
        if isinstance(helper_or_config, Mapping):
            return DataHelper(dict(helper_or_config))
        return DataHelper(helper_or_config)

    @staticmethod
    def _normalize_date(value: str | dt.date) -> dt.date:
        if isinstance(value, dt.date):
            return value
        return dt.date.fromisoformat(value)

    @staticmethod
    def _require_bounds(start: str | None, end: str | None) -> tuple[str, str]:
        if start is None or end is None:
            raise RuntimeError("Hybrid load plan is missing date boundaries.")
        return start, end

    def __enter__(self) -> HybridDataset:
        self.historical.__enter__()
        self.live.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.historical.__exit__(exc_type, exc_val, exc_tb)
        self.live.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> HybridDataset:
        await self.historical.__aenter__()
        await self.live.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.historical.__aexit__(exc_type, exc_val, exc_tb)
        await self.live.__aexit__(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        self.historical.close()
        self.live.close()

    async def aclose(self) -> None:
        await self.historical.aclose()
        await self.live.aclose()

    @property
    def dask(self) -> _HybridEngineBoundDataset:
        return _HybridEngineBoundDataset(self, return_type="dask", execution_mode="lazy")

    @property
    def pandas(self) -> _HybridEngineBoundDataset:
        return _HybridEngineBoundDataset(self, return_type="pandas", execution_mode="eager")

    @property
    def polars(self) -> _HybridEngineBoundDataset:
        return _HybridEngineBoundDataset(self, return_type="polars", execution_mode="eager")

    @property
    def arrow(self) -> _HybridEngineBoundDataset:
        return _HybridEngineBoundDataset(self, return_type="arrow", execution_mode="eager")

    def load(
        self,
        *,
        start: str,
        end: str,
        source: HybridSource = "auto",
        **options: Any,
    ) -> FrameResult:
        plan = self._resolve_plan(start=start, end=end, source=source)
        if plan.source == "historical":
            historical_start, historical_end = self._require_bounds(
                plan.historical_start,
                plan.historical_end,
            )
            return self.historical.load_period(
                self.date_field,
                historical_start,
                historical_end,
                **options,
            )
        if plan.source == "live":
            live_start, live_end = self._require_bounds(plan.live_start, plan.live_end)
            return self.live.load_period(self.date_field, live_start, live_end, **options)
        return self._load_mixed_sync(plan, options)

    async def aload(
        self,
        *,
        start: str,
        end: str,
        source: HybridSource = "auto",
        **options: Any,
    ) -> FrameResult:
        plan = self._resolve_plan(start=start, end=end, source=source)
        if plan.source == "historical":
            historical_start, historical_end = self._require_bounds(
                plan.historical_start,
                plan.historical_end,
            )
            return await self.historical.aload_period(
                self.date_field,
                historical_start,
                historical_end,
                **options,
            )
        if plan.source == "live":
            live_start, live_end = self._require_bounds(plan.live_start, plan.live_end)
            return await self.live.aload_period(self.date_field, live_start, live_end, **options)
        return await self._load_mixed_async(plan, options)

    def _resolve_plan(self, *, start: str, end: str, source: HybridSource) -> _LoadPlan:
        if source not in {"auto", "historical", "live"}:
            raise ValueError("source must be one of: 'auto', 'historical', 'live'.")

        start_date = dt.date.fromisoformat(start)
        end_date = dt.date.fromisoformat(end)
        if end_date < start_date:
            raise ValueError("end must be greater than or equal to start.")

        if source == "historical":
            return _LoadPlan(source="historical", historical_start=start, historical_end=end)
        if source == "live":
            return _LoadPlan(source="live", live_start=start, live_end=end)

        if end_date < self.split_date:
            return _LoadPlan(source="historical", historical_start=start, historical_end=end)
        if start_date >= self.split_date:
            return _LoadPlan(source="live", live_start=start, live_end=end)

        historical_end = (self.split_date - dt.timedelta(days=1)).isoformat()
        live_start = self.split_date.isoformat()
        return _LoadPlan(
            source="auto",
            historical_start=start,
            historical_end=historical_end,
            live_start=live_start,
            live_end=end,
        )

    @staticmethod
    def _resolved_mixed_return_type(options: Mapping[str, Any]) -> ResolvedReturnType:
        requested = options.get("return_type")
        if requested is None or requested == "auto":
            return "dask"
        if requested in {"pandas", "arrow", "dask", "polars"}:
            return cast(ResolvedReturnType, requested)
        raise ValueError(f"Unsupported return_type for HybridDataset mixed load: {requested!r}")

    def _branch_options(
        self,
        options: Mapping[str, Any],
        *,
        return_type: ResolvedReturnType,
    ) -> tuple[dict[str, Any], bool, bool]:
        branch_options = dict(options)
        persist = bool(branch_options.pop("persist", False))
        resilient = bool(branch_options.pop("resilient", False))
        branch_options["return_type"] = return_type
        if return_type == "dask":
            branch_options.setdefault("execution_mode", "lazy")
        elif "execution_mode" not in branch_options:
            branch_options["execution_mode"] = "eager"
        return branch_options, persist, resilient

    def _combine_mixed_frames(
        self,
        frames: list[FrameResult],
        *,
        return_type: ResolvedReturnType,
        persist: bool,
        resilient: bool,
    ) -> FrameResult:
        if return_type == "dask":
            combined = dd.concat([_to_dask(frame) for frame in frames], ignore_unknown_divisions=True)
        elif return_type == "pandas":
            combined = pd.concat([_to_pandas(frame) for frame in frames], ignore_index=True)
        elif return_type == "arrow":
            combined = pa.concat_tables([_to_arrow(frame) for frame in frames])
        elif return_type == "polars":
            combined = pl.concat([_to_polars(frame) for frame in frames], how="vertical_relaxed")
        else:
            raise ValueError(f"Unsupported return_type for HybridDataset mixed load: {return_type!r}")

        if persist and isinstance(combined, dd.DataFrame):
            if resilient:
                return safe_persist(combined)
            return combined.persist()
        return combined

    def _load_mixed_sync(self, plan: _LoadPlan, options: Mapping[str, Any]) -> FrameResult:
        return_type = self._resolved_mixed_return_type(options)
        branch_options, persist, resilient = self._branch_options(options, return_type=return_type)
        historical_start, historical_end = self._require_bounds(plan.historical_start, plan.historical_end)
        live_start, live_end = self._require_bounds(plan.live_start, plan.live_end)
        historical_frame = self.historical.load_period(
            self.date_field,
            historical_start,
            historical_end,
            **branch_options,
        )
        live_frame = self.live.load_period(
            self.date_field,
            live_start,
            live_end,
            **branch_options,
        )
        return self._combine_mixed_frames(
            [historical_frame, live_frame],
            return_type=return_type,
            persist=persist,
            resilient=resilient,
        )

    async def _load_mixed_async(self, plan: _LoadPlan, options: Mapping[str, Any]) -> FrameResult:
        return_type = self._resolved_mixed_return_type(options)
        branch_options, persist, resilient = self._branch_options(options, return_type=return_type)
        historical_start, historical_end = self._require_bounds(plan.historical_start, plan.historical_end)
        live_start, live_end = self._require_bounds(plan.live_start, plan.live_end)
        historical_task = self.historical.aload_period(
            self.date_field,
            historical_start,
            historical_end,
            **branch_options,
        )
        live_task = self.live.aload_period(
            self.date_field,
            live_start,
            live_end,
            **branch_options,
        )
        historical_frame, live_frame = await asyncio.gather(historical_task, live_task)
        return self._combine_mixed_frames(
            [historical_frame, live_frame],
            return_type=return_type,
            persist=persist,
            resilient=resilient,
        )


__all__ = ["HybridDataset"]

