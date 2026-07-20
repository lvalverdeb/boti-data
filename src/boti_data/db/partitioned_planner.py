from __future__ import annotations

import math
from typing import Any

from sqlalchemy.sql import Select

from boti_data.db.partitioned_planner_async import _async_plan_offset, _async_plan_range
from boti_data.db.partitioned_planning_math import (
    build_numeric_range_bounds,
    build_range_bounds,
    build_temporal_range_bounds,
    restore_temporal_bound,
)
from boti_data.db.partitioned_statements import (
    base_statement,
    bounds_statement,
    count_statement,
    count_up_to_statement,
    infer_meta_dtypes,
)
from boti_data.db.partitioned_types import (
    SqlOffsetPlanContext,
    SqlPartitionedLoadRequest,
    SqlPartitionPlan,
    SqlPartitionSpec,
    _prepare_offset_plan_context,
    _resolve_model_column,
)
from boti_data.db.sql_resource import SqlDatabaseResource
from boti_data.filters import FilterHandler


class SqlPartitionPlanner:
    """Plan partitioned SQL reads without worker execution concerns."""

    def __init__(self, resource: SqlDatabaseResource) -> None:
        self.resource = resource

    @staticmethod
    def _make_plan(
        request: SqlPartitionedLoadRequest,
        total_rows: int,
        partitions: tuple[SqlPartitionSpec, ...],
        meta_dtypes: dict[str, str],
    ) -> SqlPartitionPlan:
        return SqlPartitionPlan(
            total_rows=total_rows,
            strategy=request.partition_strategy,
            partitions=partitions,
            meta_dtypes=meta_dtypes,
        )

    @staticmethod
    def _single_fetch_limit(request: SqlPartitionedLoadRequest) -> int:
        if request.single_fetch_threshold is not None:
            return max(request.chunk_size, request.single_fetch_threshold)
        return request.chunk_size

    def _count_total_rows(
        self, request: SqlPartitionedLoadRequest, statement: Select[Any], single_fetch_limit: int
    ) -> int:
        if request.estimated_rows is not None and request.estimated_rows <= single_fetch_limit:
            total_rows = request.estimated_rows
        else:
            total_rows = self.count_rows_up_to(statement, single_fetch_limit + 1)
        if total_rows > single_fetch_limit:
            total_rows = self.count_rows(statement)
        if request.limit is not None:
            total_rows = min(total_rows, request.limit)
        return total_rows

    def _dispatch_partitions(
        self,
        request: SqlPartitionedLoadRequest,
        statement: Select[Any],
        total_rows: int,
        single_fetch_limit: int,
    ) -> tuple[SqlPartitionSpec, ...]:
        if total_rows <= single_fetch_limit:
            # Return a single partition — call offset planner so it
            # compiles the full-statement spec (same as its own ≤chunk_size path).
            return self.plan_offset_partitions(
                statement=statement,
                model=request.model,
                order_column=request.order_column,
                total_rows=total_rows,
                chunk_size=total_rows,
                limit=request.limit,
            )
        if request.partition_strategy == "range":
            return self.plan_range_partitions(
                statement=statement,
                model=request.model,
                partition_column=request.partition_column or "",
                total_rows=total_rows,
                chunk_size=request.chunk_size,
            )
        return self.plan_offset_partitions(
            statement=statement,
            model=request.model,
            order_column=request.order_column,
            total_rows=total_rows,
            chunk_size=request.chunk_size,
            limit=request.limit,
        )

    def plan_request(self, request: SqlPartitionedLoadRequest) -> SqlPartitionPlan:
        statement = self.prepare_statement(request)
        meta_dtypes = self.infer_meta_dtypes(statement)
        single_fetch_limit = self._single_fetch_limit(request)
        total_rows = self._count_total_rows(request, statement, single_fetch_limit)
        if total_rows <= 0:
            return self._make_plan(request, 0, (), meta_dtypes)
        partitions = self._dispatch_partitions(request, statement, total_rows, single_fetch_limit)
        return self._make_plan(request, total_rows, partitions, meta_dtypes)

    def prepare_statement(self, request: SqlPartitionedLoadRequest) -> Select[Any]:
        statement = request.statement
        if request.params:
            statement = statement.params(**request.params)
        if request.filters:
            handler = FilterHandler(
                backend="sqlalchemy",
                logger=self.resource.logger,
                debug=self.resource.debug,
            )
            statement = handler.apply_filters(
                statement,
                model=request.model,
                filters=request.filters,
            )
        return statement

    # Not a copy-pasted twin: shares count_statement() with _async_count_rows();
    # _async_count_rows() takes an externally-managed conn because it runs
    # concurrently (asyncio.gather) with _async_get_filtered_bounds() in
    # _async_plan_range, while this owns its own connection — a genuine
    # signature difference, not just sync/async syntax.
    # spaghetti-ignore[sync-async-duplication]
    def count_rows(self, statement: Select[Any]) -> int:
        statement_to_run = count_statement(statement)
        with self.resource.engine.connect() as conn:
            return int(conn.execute(statement_to_run).scalar_one())

    # Not a copy-pasted twin: same reasoning as count_rows() above, vs.
    # _async_count_rows_up_to()'s externally-managed conn used inside
    # _async_plan_offset's connection scoping.
    # spaghetti-ignore[sync-async-duplication]
    def count_rows_up_to(self, statement: Select[Any], max_rows: int) -> int:
        statement_to_run = count_up_to_statement(statement, max_rows)
        with self.resource.engine.connect() as conn:
            return int(conn.execute(statement_to_run).scalar_one())

    async def async_plan_request(
        self, request: SqlPartitionedLoadRequest, engine: Any
    ) -> SqlPartitionPlan:
        """Plan a partitioned load; opens its own async connections.

        For ``range`` strategy the COUNT and MIN/MAX planning queries are issued
        in parallel over two independent connections, halving planning latency.
        """
        statement = self.prepare_statement(request)
        meta_dtypes = self.infer_meta_dtypes(statement)

        if request.estimated_rows is not None and request.estimated_rows <= request.chunk_size:
            return self._plan_small_estimated_result(request, statement, meta_dtypes)

        if request.partition_strategy == "range":
            return await _async_plan_range(self, request, statement, meta_dtypes, engine)
        return await _async_plan_offset(self, request, statement, meta_dtypes, engine)

    def _plan_small_estimated_result(
        self,
        request: SqlPartitionedLoadRequest,
        statement: Select[Any],
        meta_dtypes: dict[str, str],
    ) -> SqlPartitionPlan:
        """Build a single-partition plan when the estimated row count is small.

        Split out of async_plan_request(): this branch needs no engine access
        (no COUNT/MIN-MAX queries), so it stays synchronous.
        """
        total_rows = request.estimated_rows
        if request.limit is not None:
            total_rows = min(total_rows, request.limit)
        if total_rows <= 0:
            return self._make_plan(request, 0, (), meta_dtypes)
        base = self.base_statement(statement)
        if request.partition_strategy == "offset":
            base = base.limit(request.chunk_size).offset(0)
        return self._make_plan(request, total_rows, (self.compile_partition(base),), meta_dtypes)

    async def _async_compute_ordering_bounds(
        self,
        base_statement: Select[Any],
        ordering_column: Any,
        engine: Any,
    ) -> tuple[Any, Any]:
        statement_to_run = bounds_statement(base_statement, ordering_column)
        async with engine.connect() as conn:
            result = await conn.execute(statement_to_run)
            return result.one()

    async def _async_count_rows(self, statement: Select[Any], conn: Any) -> int:
        statement_to_run = count_statement(statement)
        result = await conn.execute(statement_to_run)
        return int(result.scalar_one())

    async def _async_count_rows_up_to(
        self, statement: Select[Any], max_rows: int, conn: Any
    ) -> int:
        statement_to_run = count_up_to_statement(statement, max_rows)
        result = await conn.execute(statement_to_run)
        return int(result.scalar_one())

    # Not a copy-pasted twin: both already share bounds_statement()/base_statement();
    # this takes an externally-managed conn for concurrent use in
    # _async_plan_range's asyncio.gather, while get_filtered_bounds() owns its
    # own connection — a genuine signature difference, not just sync/async syntax.
    # spaghetti-ignore[sync-async-duplication]
    async def _async_get_filtered_bounds(
        self,
        statement: Select[Any],
        partitioning_column: Any,
        conn: Any,
    ) -> tuple[Any, Any]:
        statement_to_run = bounds_statement(base_statement(statement), partitioning_column)
        result = await conn.execute(statement_to_run)
        minimum, maximum = result.one()
        return minimum, maximum

    def plan_offset_partitions(
        self,
        *,
        statement: Select[Any],
        model: Any,
        order_column: str | None,
        total_rows: int,
        chunk_size: int,
        limit: int | None = None,
    ) -> tuple[SqlPartitionSpec, ...]:
        context = _prepare_offset_plan_context(
            statement=statement,
            model=model,
            order_column=order_column,
            total_rows=total_rows,
            chunk_size=chunk_size,
            limit=limit,
        )

        # Keyset partitioning divides the ordering column's *value range*, so its
        # partitions always span the full result and cannot honor a row ``limit``.
        # When a limit is active, fall back to LIMIT/OFFSET partitioning, whose
        # partitions sum to exactly ``total_rows`` (already clamped to the limit).
        if context.keyset_eligible:
            lower_bound, upper_bound = self._compute_ordering_bounds(
                context.base_statement,
                context.ordering_column,
            )
            if lower_bound is not None and upper_bound is not None:
                return self._plan_keyset_partitions(context, lower_bound, upper_bound)

        return self._plan_offset_partitions_legacy(context)

    def _compute_ordering_bounds(
        self,
        base_statement: Select[Any],
        ordering_column: Any,
    ) -> tuple[Any, Any]:
        statement_to_run = bounds_statement(base_statement, ordering_column)
        with self.resource.engine.connect() as conn:
            return conn.execute(statement_to_run).one()

    def _plan_keyset_partitions(
        self,
        context: SqlOffsetPlanContext,
        lower_bound: Any,
        upper_bound: Any,
    ) -> tuple[SqlPartitionSpec, ...]:
        if lower_bound is None or upper_bound is None:
            return (self.compile_partition(context.base_statement.limit(context.chunk_size)),)
        if lower_bound == upper_bound:
            return (
                self.compile_partition(
                    context.base_statement.where(context.ordering_column >= lower_bound),
                ),
            )

        target_partitions = max(1, math.ceil(context.total_rows / context.chunk_size))
        target_partitions = min(target_partitions, context.total_rows)

        bounds = build_numeric_range_bounds(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target_partitions=target_partitions,
        )

        partitions: list[SqlPartitionSpec] = []
        for lower, upper in bounds:
            stmt = context.base_statement.where(context.ordering_column >= lower)
            if upper is not None:
                stmt = stmt.where(context.ordering_column < upper)
            partitions.append(self.compile_partition(stmt))

        return tuple(partitions)

    def _plan_offset_partitions_legacy(
        self,
        context: SqlOffsetPlanContext,
    ) -> tuple[SqlPartitionSpec, ...]:
        partitions: list[SqlPartitionSpec] = []
        for offset in range(0, context.total_rows, context.chunk_size):
            partition_limit = min(context.chunk_size, context.total_rows - offset)
            partitions.append(
                self.compile_partition(context.base_statement.limit(partition_limit).offset(offset))
            )
        return tuple(partitions)

    def plan_range_partitions(
        self,
        *,
        statement: Select[Any],
        model: Any,
        partition_column: str,
        total_rows: int,
        chunk_size: int,
    ) -> tuple[SqlPartitionSpec, ...]:
        partitioning_column = _resolve_model_column(model, partition_column)
        lower_bound, upper_bound = self.get_filtered_bounds(
            statement,
            partitioning_column,
        )
        if lower_bound is None or upper_bound is None:
            return ()

        target_partitions = max(1, math.ceil(total_rows / chunk_size))
        target_partitions = min(target_partitions, total_rows)
        bounds = build_range_bounds(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target_partitions=target_partitions,
        )

        resolved_base_statement = base_statement(statement)
        partitions: list[SqlPartitionSpec] = []
        for lower, upper in bounds:
            bounded_statement = resolved_base_statement.where(partitioning_column >= lower)
            if upper is not None:
                bounded_statement = bounded_statement.where(partitioning_column < upper)
            partitions.append(self.compile_partition(bounded_statement))
        return tuple(partitions)

    def get_filtered_bounds(
        self,
        statement: Select[Any],
        partitioning_column: Any,
    ) -> tuple[Any, Any]:
        statement_to_run = bounds_statement(base_statement(statement), partitioning_column)
        with self.resource.engine.connect() as conn:
            minimum, maximum = conn.execute(statement_to_run).one()
        return minimum, maximum

    base_statement = staticmethod(base_statement)

    def compile_partition(self, statement: Select[Any]) -> SqlPartitionSpec:
        compiled = statement.compile(
            dialect=self.resource.engine.dialect,
            compile_kwargs={"render_postcompile": True},
        )
        positiontup = getattr(compiled, "positiontup", None)
        if positiontup:
            params = tuple(compiled.params[name] for name in positiontup)
        else:
            params = compiled.params or None
        return SqlPartitionSpec(sql=str(compiled), params=params)

    infer_meta_dtypes = staticmethod(infer_meta_dtypes)

    build_range_bounds = staticmethod(build_range_bounds)
    build_numeric_range_bounds = staticmethod(build_numeric_range_bounds)
    build_temporal_range_bounds = staticmethod(build_temporal_range_bounds)
    restore_temporal_bound = staticmethod(restore_temporal_bound)
