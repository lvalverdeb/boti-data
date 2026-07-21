"""Shared types, frame helpers, and the engine-bound view for HybridDataset.

Split out of hybrid.py purely for line-count headroom: these are either pure
functions/dataclasses with no dependency on HybridDataset's own state, or a
thin view class that only calls back into HybridDataset's public load()/
aload() API. Nothing outside hybrid.py references any of this directly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa

from boti_data.datacube import DatacubeConfig
from boti_data.db import SqlDatabaseConfig
from boti_data.gateway.frame_strategies import _to_arrow
from boti_data.parquet.resource import ParquetDataConfig

if TYPE_CHECKING:
    from boti_data.dataset.hybrid import HybridDataset

HybridSource = Literal["auto", "historical", "live"]
ResolvedReturnType = Literal["pandas", "arrow", "dask", "polars"]
BackendConfig = SqlDatabaseConfig | ParquetDataConfig | DatacubeConfig
FrameResult = pd.DataFrame | dd.DataFrame | pa.Table | pl.DataFrame


_TO_PANDAS_CONVERTERS: list[tuple[type, Callable[[FrameResult], pd.DataFrame]]] = [
    (pd.DataFrame, lambda frame: frame),
    (dd.DataFrame, lambda frame: frame.compute()),
    (pa.Table, lambda frame: frame.to_pandas()),
    (pl.DataFrame, lambda frame: frame.to_pandas()),
]


def _to_pandas(frame: FrameResult) -> pd.DataFrame:
    for frame_type, convert in _TO_PANDAS_CONVERTERS:
        if isinstance(frame, frame_type):
            return convert(frame)
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


@dataclass(slots=True)
class _MixedLoadPrep:
    return_type: ResolvedReturnType
    branch_options: dict[str, Any]
    persist: bool
    resilient: bool
    historical_start: Any
    historical_end: Any
    live_start: Any
    live_end: Any


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

    # Not a copy-pasted twin: pure pass-through to the already-split
    # HybridDataset.load()/aload() twins, using the shared _bind_options().
    # spaghetti-ignore[sync-async-duplication]: see above
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
