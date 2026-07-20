from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Mapping
from typing import Any, cast

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
from boti.core.lifecycle import LifecycleCore
from boti.core.lifecycle_pickle import PicklableLifecycleCoreMixin
from boti_dask import safe_persist

from boti_data.dataset.hybrid_support import (
    BackendConfig,
    FrameResult,
    HybridSource,
    ResolvedReturnType,
    _HybridEngineBoundDataset,
    _LoadPlan,
    _MixedLoadPrep,
    _to_arrow,
    _to_dask,
    _to_pandas,
    _to_polars,
)
from boti_data.helper import DataHelper

_MIXED_CONCAT_FNS: dict[ResolvedReturnType, Any] = {
    "dask": lambda frames: dd.concat(
        [_to_dask(frame) for frame in frames], ignore_unknown_divisions=True
    ),
    "pandas": lambda frames: pd.concat([_to_pandas(frame) for frame in frames], ignore_index=True),
    "arrow": lambda frames: pa.concat_tables([_to_arrow(frame) for frame in frames]),
    "polars": lambda frames: pl.concat(
        [_to_polars(frame) for frame in frames], how="vertical_relaxed"
    ),
}


def _persist_if_requested(combined: FrameResult, *, persist: bool, resilient: bool) -> FrameResult:
    if not persist or not isinstance(combined, dd.DataFrame):
        return combined
    if resilient:
        return safe_persist(combined)
    return combined.persist()


class HybridDataset(PicklableLifecycleCoreMixin, LifecycleCore):
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
        super().__init__()

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
        super().__enter__()
        self.historical.__enter__()
        self.live.__enter__()
        return self

    async def __aenter__(self) -> HybridDataset:
        await super().__aenter__()
        await self.historical.__aenter__()
        await self.live.__aenter__()
        return self

    def _cleanup(self) -> None:
        # try/finally so a failure closing `historical` can't leak `live` —
        # the original hand-written close() called both unconditionally in
        # sequence, so a historical.close() exception left live still open.
        try:
            self.historical.close()
        finally:
            self.live.close()

    async def _acleanup(self) -> None:
        try:
            await self.historical.aclose()
        finally:
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

    # Not a copy-pasted twin: both already share _resolve_plan() and dispatch to
    # the already-split historical/live load_period()/aload_period() twins and
    # _load_mixed_sync()/_load_mixed_async() below — nothing further to extract.
    # spaghetti-ignore[sync-async-duplication]
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

        if source == "historical" or (source == "auto" and end_date < self.split_date):
            return _LoadPlan(source="historical", historical_start=start, historical_end=end)
        if source == "live" or (source == "auto" and start_date >= self.split_date):
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
        concat_fn = _MIXED_CONCAT_FNS.get(return_type)
        if concat_fn is None:
            raise ValueError(
                f"Unsupported return_type for HybridDataset mixed load: {return_type!r}"
            )
        combined = concat_fn(frames)
        return _persist_if_requested(combined, persist=persist, resilient=resilient)

    def _prepare_mixed_load(self, plan: _LoadPlan, options: Mapping[str, Any]) -> _MixedLoadPrep:
        return_type = self._resolved_mixed_return_type(options)
        branch_options, persist, resilient = self._branch_options(options, return_type=return_type)
        historical_start, historical_end = self._require_bounds(
            plan.historical_start, plan.historical_end
        )
        live_start, live_end = self._require_bounds(plan.live_start, plan.live_end)
        return _MixedLoadPrep(
            return_type=return_type,
            branch_options=branch_options,
            persist=persist,
            resilient=resilient,
            historical_start=historical_start,
            historical_end=historical_end,
            live_start=live_start,
            live_end=live_end,
        )

    def _combine_from_prep(
        self, prep: _MixedLoadPrep, historical_frame: FrameResult, live_frame: FrameResult
    ) -> FrameResult:
        return self._combine_mixed_frames(
            [historical_frame, live_frame],
            return_type=prep.return_type,
            persist=prep.persist,
            resilient=prep.resilient,
        )

    # Not a copy-pasted twin: shared prep (_prepare_mixed_load) and shared
    # combine (_combine_from_prep) are already extracted above; the remaining
    # difference is a genuine behavioral one — _load_mixed_async fetches
    # historical+live concurrently via asyncio.gather, _load_mixed_sync is
    # sequential.
    # spaghetti-ignore[sync-async-duplication]
    def _load_mixed_sync(self, plan: _LoadPlan, options: Mapping[str, Any]) -> FrameResult:
        prep = self._prepare_mixed_load(plan, options)
        historical_frame = self.historical.load_period(
            self.date_field,
            prep.historical_start,
            prep.historical_end,
            **prep.branch_options,
        )
        live_frame = self.live.load_period(
            self.date_field,
            prep.live_start,
            prep.live_end,
            **prep.branch_options,
        )
        return self._combine_from_prep(prep, historical_frame, live_frame)

    async def _load_mixed_async(self, plan: _LoadPlan, options: Mapping[str, Any]) -> FrameResult:
        prep = self._prepare_mixed_load(plan, options)
        historical_task = self.historical.aload_period(
            self.date_field,
            prep.historical_start,
            prep.historical_end,
            **prep.branch_options,
        )
        live_task = self.live.aload_period(
            self.date_field,
            prep.live_start,
            prep.live_end,
            **prep.branch_options,
        )
        historical_frame, live_frame = await asyncio.gather(historical_task, live_task)
        return self._combine_from_prep(prep, historical_frame, live_frame)


__all__ = ["HybridDataset"]
