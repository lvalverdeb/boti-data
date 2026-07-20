"""Async range/offset planning orchestration for SqlPartitionPlanner.

Split out of partitioned_planner.py purely for line-count headroom: these
are orchestration methods that only need the planner instance to reach back
into its own private async helpers, so they move here as free functions
taking the planner explicitly, mirroring the gateway/_backend_strategies.py
split precedent (functions taking a ``ctx`` instead of ``self``).
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any

from sqlalchemy.sql import Select

from boti_data.db.partitioned_planning_math import build_range_bounds
from boti_data.db.partitioned_statements import base_statement
from boti_data.db.partitioned_types import (
    SqlPartitionedLoadRequest,
    SqlPartitionPlan,
    SqlPartitionSpec,
    _prepare_offset_plan_context,
    _resolve_model_column,
)

if TYPE_CHECKING:
    from boti_data.db.partitioned_planner import SqlPartitionPlanner


async def _async_plan_range(
    planner: SqlPartitionPlanner,
    request: SqlPartitionedLoadRequest,
    statement: Select[Any],
    meta_dtypes: dict[str, str],
    engine: Any,
) -> SqlPartitionPlan:
    partitioning_column = _resolve_model_column(request.model, request.partition_column or "")
    async with engine.connect() as count_conn, engine.connect() as bounds_conn:
        total_rows, (lower_bound, upper_bound) = await asyncio.gather(
            planner._async_count_rows(statement, count_conn),
            planner._async_get_filtered_bounds(statement, partitioning_column, bounds_conn),
        )
    if request.limit is not None:
        total_rows = min(total_rows, request.limit)
    if total_rows <= 0 or lower_bound is None or upper_bound is None:
        return planner._make_plan(request, max(0, total_rows), (), meta_dtypes)
    target_partitions = max(1, math.ceil(total_rows / request.chunk_size))
    target_partitions = min(target_partitions, total_rows)
    bounds = build_range_bounds(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        target_partitions=target_partitions,
    )
    resolved_base_statement = base_statement(statement)
    parts: list[SqlPartitionSpec] = []
    for lower, upper in bounds:
        bounded = resolved_base_statement.where(partitioning_column >= lower)
        if upper is not None:
            bounded = bounded.where(partitioning_column < upper)
        parts.append(planner.compile_partition(bounded))
    partitions: tuple[SqlPartitionSpec, ...] = tuple(parts)
    return planner._make_plan(request, total_rows, partitions, meta_dtypes)


async def _async_plan_offset(
    planner: SqlPartitionPlanner,
    request: SqlPartitionedLoadRequest,
    statement: Select[Any],
    meta_dtypes: dict[str, str],
    engine: Any,
) -> SqlPartitionPlan:
    async with engine.connect() as conn:
        total_rows = await planner._async_count_rows_up_to(
            statement,
            request.chunk_size + 1,
            conn,
        )
    if total_rows > request.chunk_size:
        async with engine.connect() as conn:
            total_rows = await planner._async_count_rows(statement, conn)
    if request.limit is not None:
        total_rows = min(total_rows, request.limit)
    if total_rows <= 0:
        return planner._make_plan(request, 0, (), meta_dtypes)

    context = _prepare_offset_plan_context(
        statement=statement,
        model=request.model,
        order_column=request.order_column,
        total_rows=total_rows,
        chunk_size=request.chunk_size,
        limit=request.limit,
    )

    # Keyset partitioning divides the ordering column's value range and cannot
    # honor a row ``limit``; when a limit is active, use LIMIT/OFFSET below.
    if context.keyset_eligible:
        lower_bound, upper_bound = await planner._async_compute_ordering_bounds(
            context.base_statement,
            context.ordering_column,
            engine,
        )
        if lower_bound is not None and upper_bound is not None:
            partitions = planner._plan_keyset_partitions(context, lower_bound, upper_bound)
            return planner._make_plan(request, total_rows, partitions, meta_dtypes)

    partitions = planner._plan_offset_partitions_legacy(context)
    return planner._make_plan(request, total_rows, partitions, meta_dtypes)
