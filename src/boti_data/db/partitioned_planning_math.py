"""Pure range-bucketing math for SqlPartitionPlanner's range/keyset partitioning."""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal
from typing import Any

import pandas as pd


def build_range_bounds(
    *,
    lower_bound: Any,
    upper_bound: Any,
    target_partitions: int,
) -> list[tuple[Any, Any]]:
    if isinstance(lower_bound, bool) or isinstance(upper_bound, bool):
        raise ValueError("range partitioning does not support boolean partition columns.")

    if isinstance(lower_bound, (int, float, Decimal)) and isinstance(
        upper_bound, (int, float, Decimal)
    ):
        return build_numeric_range_bounds(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target_partitions=target_partitions,
        )

    if isinstance(lower_bound, (dt.date, dt.datetime)) and isinstance(
        upper_bound,
        (dt.date, dt.datetime),
    ):
        return build_temporal_range_bounds(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target_partitions=target_partitions,
        )

    raise ValueError(
        "range partitioning currently supports numeric and temporal partition columns only."
    )


def build_numeric_range_bounds(
    *,
    lower_bound: Any,
    upper_bound: Any,
    target_partitions: int,
) -> list[tuple[Any, Any]]:
    span = upper_bound - lower_bound
    step = max(1, math.ceil((span + 1) / target_partitions))
    # Pre-allocate: actual count is at most target_partitions + 1
    partitions: list[tuple[Any, Any]] = [None] * (target_partitions + 1)  # type: ignore[list-item]
    idx = 0
    current = lower_bound
    while current <= upper_bound:
        next_value = current + step
        partitions[idx] = (current, next_value)
        idx += 1
        current = next_value
    return partitions[:idx]


def build_temporal_range_bounds(
    *,
    lower_bound: Any,
    upper_bound: Any,
    target_partitions: int,
) -> list[tuple[Any, Any]]:
    lower_ts = pd.Timestamp(lower_bound)
    upper_ts = pd.Timestamp(upper_bound)
    span_ns = max(1, upper_ts.value - lower_ts.value + 1)
    step_ns = max(1, math.ceil(span_ns / target_partitions))
    step = pd.Timedelta(step_ns, unit="ns")
    if isinstance(lower_bound, dt.date) and not isinstance(lower_bound, dt.datetime):
        step = max(step, pd.Timedelta(days=1))

    # Pre-allocate: actual count is at most target_partitions + 1
    partitions: list[tuple[Any, Any]] = [None] * (target_partitions + 1)  # type: ignore[list-item]
    idx = 0
    current = lower_ts
    while current <= upper_ts:
        next_value = current + step
        partitions[idx] = (
            restore_temporal_bound(current, lower_bound),
            restore_temporal_bound(next_value, lower_bound),
        )
        idx += 1
        current = next_value
    return partitions[:idx]


def restore_temporal_bound(value: pd.Timestamp, template: Any) -> Any:
    if isinstance(template, dt.datetime):
        return value.to_pydatetime()
    if isinstance(template, dt.date):
        return value.date()
    return value
