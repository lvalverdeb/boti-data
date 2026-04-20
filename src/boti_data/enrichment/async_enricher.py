from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol, Union

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa

from boti_data.enrichment.specs import AttachmentSpec

FrameResult = Union[pd.DataFrame, dd.DataFrame, pa.Table, pl.DataFrame]


def _to_dask_frame(frame: FrameResult) -> dd.DataFrame:
    if isinstance(frame, dd.DataFrame):
        return frame
    if isinstance(frame, pd.DataFrame):
        return dd.from_pandas(frame, npartitions=max(1, len(frame) or 1))
    if isinstance(frame, pa.Table):
        return dd.from_pandas(frame.to_pandas(), npartitions=1)
    if isinstance(frame, pl.DataFrame):
        return dd.from_pandas(frame.to_pandas(), npartitions=1)
    raise TypeError(f"Unsupported frame type for enrichment: {type(frame)!r}")


class FrameEnricher(Protocol):
    def enrich(self, base_frame: FrameResult, *, cols: Sequence[str] | None = None) -> FrameResult: ...

    async def aenrich(
        self,
        base_frame: FrameResult,
        *,
        cols: Sequence[str] | None = None,
    ) -> FrameResult: ...


class AsyncFrameEnricher:
    """Spec-driven async frame enricher with bounded unique extraction and safe merges."""

    def __init__(
        self,
        specs: Sequence[AttachmentSpec],
        *,
        default_max_unique_values: int = 5000,
        max_concurrency: int = 4,
    ) -> None:
        if default_max_unique_values < 1:
            raise ValueError("default_max_unique_values must be >= 1.")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1.")
        self.specs = list(specs)
        self.default_max_unique_values = default_max_unique_values
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def enrich(self, base_frame: FrameResult, *, cols: Sequence[str] | None = None) -> FrameResult:
        return asyncio.run(self.aenrich(base_frame, cols=cols))

    async def aenrich(
        self,
        base_frame: FrameResult,
        *,
        cols: Sequence[str] | None = None,
    ) -> FrameResult:
        base = _to_dask_frame(base_frame)
        available_cols = set(str(col) for col in base.columns)
        requested_keys = set(cols) if cols else None

        applicable = [
            spec
            for spec in self.specs
            if spec.is_applicable(available_cols, requested_keys=requested_keys)
        ]
        if not applicable:
            return base

        attach_tasks = [self._run_attachment(base, spec) for spec in applicable]
        attachments = await asyncio.gather(*attach_tasks)

        out = base
        for spec, attached in zip(applicable, attachments):
            if attached is None:
                continue
            out = self._merge_attachment(out, _to_dask_frame(attached), spec)
        return out

    async def _run_attachment(
        self,
        base: dd.DataFrame,
        spec: AttachmentSpec,
    ) -> FrameResult | None:
        kwargs: dict[str, Any] = {}
        for column, kwarg_name in spec.col_to_kwarg.items():
            if column not in base.columns:
                return None
            values = await self._extract_unique_values(
                base[column],
                max_unique=spec.max_unique_values or self.default_max_unique_values,
                column_name=column,
            )
            if not values:
                return None
            kwargs[kwarg_name] = values

        if not kwargs:
            return None
        return await spec.attachment_fn(**kwargs)

    async def _extract_unique_values(
        self,
        series: dd.Series | pd.Series,
        *,
        max_unique: int,
        column_name: str,
    ) -> list[Any]:
        async with self._semaphore:
            if isinstance(series, pd.Series):
                uniques = series.dropna().drop_duplicates().tolist()
            else:
                # Bound extraction work: request max+1 unique values.
                uniques = await asyncio.to_thread(
                    lambda: series.dropna().drop_duplicates().head(
                        max_unique + 1,
                        npartitions=-1,
                        compute=True,
                    ).tolist()
                )
            if len(uniques) > max_unique:
                raise ValueError(
                    f"Column {column_name!r} exceeded unique-value limit ({max_unique})."
                )
            return uniques

    @staticmethod
    def _merge_attachment(
        left: dd.DataFrame,
        right: dd.DataFrame,
        spec: AttachmentSpec,
    ) -> dd.DataFrame:
        # Keep merge keys stable by promoting both sides to string.
        for l_col, r_col in zip(spec.left_on, spec.right_on):
            if l_col in left.columns:
                left = left.assign(**{l_col: left[l_col].astype("string")})
            if r_col in right.columns:
                right = right.assign(**{r_col: right[r_col].astype("string")})

        merged = left.merge(
            right,
            how="left",
            left_on=list(spec.left_on),
            right_on=list(spec.right_on),
        )
        to_drop = [col for col in spec.drop_cols if col in merged.columns]
        if to_drop:
            merged = merged.drop(columns=to_drop)
        return merged


__all__ = ["AsyncFrameEnricher", "FrameEnricher"]
