from __future__ import annotations

import asyncio
from typing import Any

import dask
import dask.dataframe as dd
import pandas as pd
import polars as pl

from boti_data.gateway.frame_strategies import get_frame_strategy


def resolve_series_filters(options: dict[str, Any]) -> dict[str, Any]:
    """Replace ``field__in=Series`` values with deduplicated plain lists.

    Pandas, Dask, and Polars Series are
    accepted.  Dask Series are synchronously computed — use
    :func:`resolve_series_filters_async` in async contexts.

    This is the pre-processing step for the distributed semi-join pattern:
    callers pass a Series of IDs as a ``field__in`` filter and the gateway
    automatically resolves it before issuing the query.  The resolved list
    is then handled by the existing chunked-IN machinery.
    """
    strategy = get_frame_strategy("dask")
    if not any(
        k.endswith("__in") and isinstance(v, (pd.Series, dd.Series, pl.Series))
        for k, v in options.items()
    ):
        return options
    resolved = dict(options)
    for key, value in options.items():
        if key.endswith("__in") and isinstance(value, (pd.Series, dd.Series, pl.Series)):
            resolved[key] = strategy.resolve_series(value)
    return resolved


def _classify_series_in_keys(options: dict[str, Any]) -> tuple[list[str], list[str]]:
    dask_keys: list[str] = []
    pandas_or_polars_keys: list[str] = []
    for key, value in options.items():
        if not key.endswith("__in"):
            continue
        if isinstance(value, dd.Series):
            dask_keys.append(key)
        elif isinstance(value, (pd.Series, pl.Series)):
            pandas_or_polars_keys.append(key)
    return dask_keys, pandas_or_polars_keys


async def resolve_series_filters_async(options: dict[str, Any]) -> dict[str, Any]:
    """Async variant of :func:`resolve_series_filters`.

    Dask Series are computed in a thread pool so the event loop is not
    blocked while Dask executes the computation graph.
    """
    dask_keys, pandas_or_polars_keys = _classify_series_in_keys(options)
    if not dask_keys and not pandas_or_polars_keys:
        return options
    resolved = dict(options)

    if dask_keys:
        computed = await asyncio.to_thread(
            dask.compute,
            *[options[key] for key in dask_keys],
        )
        for key, series in zip(dask_keys, computed):
            resolved[key] = series.dropna().unique().tolist()

    strategy = get_frame_strategy("dask")
    for key in pandas_or_polars_keys:
        resolved[key] = strategy.resolve_series(options[key])
    return resolved


def series_to_dask_key_frame(
    join_series: pd.Series | dd.Series | pl.Series,
    *,
    column_name: str,
) -> dd.DataFrame:
    if isinstance(join_series, dd.Series):
        return join_series.dropna().drop_duplicates().to_frame(name=column_name)
    if isinstance(join_series, pl.Series):
        join_series = pd.Series(join_series.to_list(), name=column_name)
    if isinstance(join_series, pd.Series):
        return dd.from_pandas(
            join_series.dropna().drop_duplicates().to_frame(name=column_name),
            npartitions=1,
        )
    raise TypeError(f"Unsupported join series type: {type(join_series)!r}")
