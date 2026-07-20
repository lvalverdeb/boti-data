from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any, Protocol, Union

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa

from boti_data.enrichment.specs import AttachmentSpec

FrameResult = Union[pd.DataFrame, dd.DataFrame, pa.Table, pl.DataFrame]


_TO_DASK_FRAME_CONVERTERS: list[tuple[type, Callable[[FrameResult], dd.DataFrame]]] = [
    (dd.DataFrame, lambda frame: frame),
    (pd.DataFrame, lambda frame: dd.from_pandas(frame, npartitions=max(1, len(frame) or 1))),
    (pa.Table, lambda frame: dd.from_pandas(frame.to_pandas(), npartitions=1)),
    (pl.DataFrame, lambda frame: dd.from_pandas(frame.to_pandas(), npartitions=1)),
]


def _to_dask_frame(frame: FrameResult) -> dd.DataFrame:
    for frame_type, convert in _TO_DASK_FRAME_CONVERTERS:
        if isinstance(frame, frame_type):
            return convert(frame)
    raise TypeError(f"Unsupported frame type for enrichment: {type(frame)!r}")


class FrameEnricher(Protocol):
    # Not a copy-pasted twin: this is a Protocol interface stub (body is `...`),
    # nothing to deduplicate.
    # spaghetti-ignore[sync-async-duplication]
    def enrich(
        self, base_frame: FrameResult, *, cols: Sequence[str] | None = None
    ) -> FrameResult: ...

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

    async def _collect_attachment_kwargs(
        self,
        base: dd.DataFrame,
        spec: AttachmentSpec,
    ) -> dict[str, Any] | None:
        """Gather attachment_fn kwargs from *base*, or None if any input is missing/empty."""
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
        return kwargs or None

    async def _run_attachment(
        self,
        base: dd.DataFrame,
        spec: AttachmentSpec,
    ) -> FrameResult | None:
        kwargs = await self._collect_attachment_kwargs(base, spec)
        if kwargs is None:
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
                deduped = series.dropna().drop_duplicates()
                uniques = deduped.tolist()
            else:
                # Bound extraction work: request max+1 unique values.
                def _extract_uniques() -> list[Any]:
                    deduped = series.dropna().drop_duplicates()
                    limited = deduped.head(max_unique + 1, npartitions=-1, compute=True)
                    return limited.tolist()

                uniques = await asyncio.to_thread(_extract_uniques)
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
