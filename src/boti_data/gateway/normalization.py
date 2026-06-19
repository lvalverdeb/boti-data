from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import dask
import dask.dataframe as dd
import pandas as pd

# Structured kwargs that belong to the load request, not filter values.
LOAD_CONTROL_KEYS = frozenset(
    {
        "sql",
        "statement",
        "model",
        "filters",
        "params",
        "allow_raw_sql",
        "raw_sql_policy",
        "limit",
        "return_type",
        "as_pandas",
        "execution_mode",
        "persist",
        "resilient",
        "dry_run",
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
        "max_concurrent_fetches",
        "raw_filters",
        "columns",
        "cube",
    }
)

# Default chunk size for splitting massive ``field__in`` filter lists before
# they hit DB parameter limits (most drivers cap at ~1 000 bind parameters).
DEFAULT_IN_CHUNK_SIZE = 900


@dataclass(frozen=True)
class NormalizedConfiguredOptions:
    control: dict[str, Any]
    filters: dict[str, Any]


_ALLOWED_BOOLEAN_KEYS = frozenset({"$and", "$or", "$not"})


def _filter_field_name(key: str) -> str:
    return key.split("__", 1)[0]


def _in_like_operator(key: str) -> bool:
    return key.endswith("__in") or key.endswith("__not_in") or key.endswith("__nin")


def validate_filter_payload(
    filters: Mapping[str, Any],
    *,
    allowed_filter_fields: set[str] | None,
    max_depth: int,
    max_conditions: int,
    max_in_values: int,
) -> None:
    """Validate filter payload complexity and allowed field keys.

    This is intentionally opt-in so existing behavior remains backward compatible.
    """

    if max_depth < 1:
        raise ValueError("max_depth must be >= 1.")
    if max_conditions < 1:
        raise ValueError("max_conditions must be >= 1.")
    if max_in_values < 1:
        raise ValueError("max_in_values must be >= 1.")

    state = {"conditions": 0}

    def _walk(node: Mapping[str, Any], depth: int) -> None:
        if depth > max_depth:
            raise ValueError(
                f"Filter payload exceeds max depth ({max_depth})."
            )
        for key, value in node.items():
            if key.startswith("$"):
                if key not in _ALLOWED_BOOLEAN_KEYS:
                    raise ValueError(f"Unsupported boolean filter operator: {key}")
                if key in {"$and", "$or"}:
                    if not isinstance(value, list):
                        raise ValueError(f"{key} filter value must be a list of mappings.")
                    for item in value:
                        if not isinstance(item, Mapping):
                            raise ValueError(f"{key} entries must be mappings.")
                        _walk(item, depth + 1)
                else:  # $not
                    if not isinstance(value, Mapping):
                        raise ValueError("$not filter value must be a mapping.")
                    _walk(value, depth + 1)
                continue

            state["conditions"] += 1
            if state["conditions"] > max_conditions:
                raise ValueError(
                    f"Filter payload exceeds max conditions ({max_conditions})."
                )

            field = _filter_field_name(key)
            if allowed_filter_fields is not None and field not in allowed_filter_fields:
                raise ValueError(
                    f"Filter field '{field}' is not allowed. "
                    "Configure allowed_filter_fields to include it."
                )

            if _in_like_operator(key) and isinstance(value, (list, tuple, set)):
                if len(value) > max_in_values:
                    raise ValueError(
                        f"Filter '{key}' exceeds max in-list size ({max_in_values})."
                    )

    _walk(filters, depth=1)


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
    strict_filter_validation: bool,
    allowed_filter_fields: set[str] | None,
    max_filter_depth: int,
    max_filter_conditions: int,
    max_in_filter_values: int,
) -> NormalizedConfiguredOptions:
    control, runtime_filters = split_control_and_filters(options)
    explicit_filters = control.pop("filters", {})
    merged_filters = {**sticky_filters, **runtime_filters, **explicit_filters}
    if strict_filter_validation:
        validate_filter_payload(
            merged_filters,
            allowed_filter_fields=allowed_filter_fields,
            max_depth=max_filter_depth,
            max_conditions=max_filter_conditions,
            max_in_values=max_in_filter_values,
        )
    if exclude and merged_filters:
        merged_filters = {"$not": merged_filters}
    return NormalizedConfiguredOptions(control=control, filters=merged_filters)


def build_partitioned_load_options(
    *,
    statement: Any,
    model: Any,
    filters: dict[str, Any],
    control: dict[str, Any],
    default_chunk_size: int | None,
) -> dict[str, Any]:
    partitioned_options: dict[str, Any] = {
        "statement": statement,
        "model": model,
        "filters": filters,
        "partitioned": True,
        "as_pandas": bool(control.get("as_pandas", False)),
    }
    limit = control.get("limit")
    if limit is not None:
        partitioned_options["limit"] = limit
    for key in (
        "max_concurrent_fetches",
        "partition_strategy",
        "partition_column",
        "order_column",
        "diagnostics",
    ):
        if key in control:
            partitioned_options[key] = control[key]
    chunk_size = control.get("chunk_size") or default_chunk_size
    if chunk_size is not None:
        partitioned_options["chunk_size"] = chunk_size
    return partitioned_options


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


def resolve_series_filters(options: dict[str, Any]) -> dict[str, Any]:
    dask_keys = [
        k for k, v in options.items()
        if k.endswith("__in") and isinstance(v, dd.Series)
    ]
    pandas_keys = [
        k for k, v in options.items()
        if k.endswith("__in") and isinstance(v, pd.Series) and not isinstance(v, dd.Series)
    ]
    if not dask_keys and not pandas_keys:
        return options

    resolved = dict(options)

    # Batch all Dask series into a single dask.compute() call to share the scheduler pass.
    if dask_keys:
        computed = dask.compute(*[options[k] for k in dask_keys])
        for key, series in zip(dask_keys, computed):
            resolved[key] = series.dropna().unique().tolist()

    for key in pandas_keys:
        resolved[key] = options[key].dropna().unique().tolist()

    return resolved


async def resolve_series_filters_async(options: dict[str, Any]) -> dict[str, Any]:
    dask_keys = [
        k
        for k, v in options.items()
        if k.endswith("__in") and isinstance(v, dd.Series)
    ]
    pandas_keys = [
        k
        for k, v in options.items()
        if k.endswith("__in") and isinstance(v, pd.Series) and not isinstance(v, dd.Series)
    ]
    if not dask_keys and not pandas_keys:
        return options

    resolved = dict(options)

    # Batch all Dask series into one scheduler pass while keeping the event loop free.
    if dask_keys:
        computed = await asyncio.to_thread(
            dask.compute,
            *[options[k] for k in dask_keys],
        )
        for key, series in zip(dask_keys, computed):
            resolved[key] = series.dropna().unique().tolist()

    # pandas Series: no I/O, resolve directly.
    for key in pandas_keys:
        resolved[key] = options[key].dropna().unique().tolist()

    return resolved
