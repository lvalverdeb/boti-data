from __future__ import annotations

import asyncio
from typing import Any

import dask
import dask.dataframe as dd
import pandas as pd

from .requests import NormalizedFilters, PartitionedLoadConfig

# Structured kwargs that belong to the load request, not filter values.
LOAD_CONTROL_KEYS = frozenset(
    {
        "sql",
        "statement",
        "model",
        "filters",
        "params",
        "limit",
        "return_type",
        "as_pandas",
        "execution_mode",
        "persist",
        "diagnostics",
        "timeout",
        "in_chunk_size",
        "in_chunk_concurrency",
        "in_chunk_strategy",
        "partitioned",
        "partition_strategy",
        "partition_column",
        "order_column",
        "chunk_size",
        "single_fetch_threshold",
        "max_concurrent_fetches",
        "raw_filters",
        "columns",
        "cube",
        "resilient",
        "dry_run",
        "allow_raw_sql",
    }
)

# Default chunk size for splitting massive ``field__in`` filter lists before
# they hit DB parameter limits (most drivers cap at ~1 000 bind parameters).
DEFAULT_IN_CHUNK_SIZE = 900


def split_control_and_filters(options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    control: dict[str, Any] = {}
    runtime_filters: dict[str, Any] = {}
    for k, v in options.items():
        if k in LOAD_CONTROL_KEYS:
            control[k] = v
        else:
            runtime_filters[k] = v
    return control, runtime_filters


def normalize_configured_filters(
    options: dict[str, Any],
    *,
    sticky_filters: dict[str, Any],
    exclude: bool,
) -> NormalizedFilters:
    control, runtime_filters = split_control_and_filters(options)
    explicit_filters = control.pop("filters", {})
    merged_filters = {**sticky_filters, **runtime_filters, **explicit_filters}
    if exclude and merged_filters:
        merged_filters = {"$not": merged_filters}
    return NormalizedFilters(control=control, filters=merged_filters)


def build_partitioned_load_options(
    *,
    statement: Any,
    model: Any,
    filters: dict[str, Any],
    control: dict[str, Any],
    default_chunk_size: int | None,
) -> PartitionedLoadConfig:
    chunk_size = control.get("chunk_size") or default_chunk_size
    return PartitionedLoadConfig(
        statement=statement,
        model=model,
        filters=filters,
        as_pandas=bool(control.get("as_pandas", False)),
        limit=control.get("limit"),
        chunk_size=chunk_size,
        single_fetch_threshold=control.get("single_fetch_threshold"),
        max_concurrent_fetches=control.get("max_concurrent_fetches"),
        partition_strategy=control.get("partition_strategy"),
        partition_column=control.get("partition_column"),
        order_column=control.get("order_column"),
        diagnostics=bool(control.get("diagnostics", False)),
    )


def prepare_period_filters(
    dt_field: str,
    start: str,
    end: str,
    **kwargs: Any,
) -> dict[str, Any]:
    start_date = pd.to_datetime(start).date()
    end_date = pd.to_datetime(end).date()
    if start_date > end_date:
        raise ValueError("'start' date cannot be later than 'end' date.")
    if start_date == end_date:
        kwargs[f"{dt_field}__exact"] = start_date
    else:
        kwargs[f"{dt_field}__gte"] = start_date
        kwargs[f"{dt_field}__lte"] = end_date
    return kwargs


def _classify_series_keys(options: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Returns (dask_keys, pandas_keys): the ``__in`` filter keys backed by a
    Dask or pandas Series, respectively (mutually exclusive)."""
    dask_keys = [k for k, v in options.items() if k.endswith("__in") and isinstance(v, dd.Series)]
    pandas_keys = [
        k
        for k, v in options.items()
        if k.endswith("__in") and isinstance(v, pd.Series) and not isinstance(v, dd.Series)
    ]
    return dask_keys, pandas_keys


def _apply_computed_dask_series(
    resolved: dict[str, Any], dask_keys: list[str], computed: tuple[Any, ...]
) -> None:
    for key, series in zip(dask_keys, computed):
        resolved[key] = series.dropna().unique().tolist()


def _resolve_pandas_series_keys(
    resolved: dict[str, Any], options: dict[str, Any], pandas_keys: list[str]
) -> None:
    # pandas Series: no I/O, resolve directly.
    for key in pandas_keys:
        resolved[key] = options[key].dropna().unique().tolist()


# Not a copy-pasted twin: both already share _classify_series_keys()/
# _apply_computed_dask_series()/_resolve_pandas_series_keys(); the remaining
# difference is dask.compute(...) vs await asyncio.to_thread(dask.compute, ...)
# — the offload boundary itself.
# spaghetti-ignore[sync-async-duplication]: see above
def resolve_series_filters(options: dict[str, Any]) -> dict[str, Any]:
    dask_keys, pandas_keys = _classify_series_keys(options)
    if not dask_keys and not pandas_keys:
        return options

    resolved = dict(options)

    # Batch all Dask series into a single dask.compute() call to share the scheduler pass.
    if dask_keys:
        computed = dask.compute(*[options[k] for k in dask_keys])
        _apply_computed_dask_series(resolved, dask_keys, computed)

    _resolve_pandas_series_keys(resolved, options, pandas_keys)
    return resolved


async def resolve_series_filters_async(options: dict[str, Any]) -> dict[str, Any]:
    dask_keys, pandas_keys = _classify_series_keys(options)
    if not dask_keys and not pandas_keys:
        return options

    resolved = dict(options)

    # Batch all Dask series into one scheduler pass while keeping the event loop free.
    if dask_keys:
        computed = await asyncio.to_thread(
            dask.compute,
            *[options[k] for k in dask_keys],
        )
        _apply_computed_dask_series(resolved, dask_keys, computed)

    _resolve_pandas_series_keys(resolved, options, pandas_keys)
    return resolved
