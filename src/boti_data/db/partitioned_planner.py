from __future__ import annotations

import asyncio
import datetime as dt
import math
from decimal import Decimal
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.sql import Select
from sqlalchemy.sql.sqltypes import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    Numeric,
    SmallInteger,
    String,
    Text,
    Time,
    Unicode,
    UnicodeText,
)

from boti_data.db.partitioned_types import (
    SqlPartitionedLoadRequest,
    SqlPartitionPlan,
    SqlPartitionSpec,
    _infer_primary_key_name,
    _resolve_model_column,
)
from boti_data.db.sql_resource import SqlDatabaseResource
from boti_data.filters import FilterHandler


def _sqlalchemy_type_to_pandas_dtype(sql_type: Any) -> str:
    if isinstance(sql_type, (SmallInteger, Integer, BigInteger)):
        return "Int64"
    if isinstance(sql_type, (Numeric, Float)):
        return "Float64"
    if isinstance(sql_type, Boolean):
        return "boolean"
    if isinstance(sql_type, (Date, DateTime)):
        return "datetime64[ns, UTC]"
    if isinstance(sql_type, (String, Text, Unicode, UnicodeText, Time)):
        return "string"
    if isinstance(sql_type, LargeBinary):
        return "object"
    return "object"


class SqlPartitionPlanner:
    """Plan partitioned SQL reads without worker execution concerns."""

    def __init__(self, resource: SqlDatabaseResource) -> None:
        self.resource = resource

    def plan_request(self, request: SqlPartitionedLoadRequest) -> SqlPartitionPlan:
        statement = self.prepare_statement(request)
        meta_dtypes = self.infer_meta_dtypes(statement)

        single_fetch_limit = request.chunk_size
        if request.single_fetch_threshold is not None:
            single_fetch_limit = max(request.chunk_size, request.single_fetch_threshold)

        if request.estimated_rows is not None and request.estimated_rows <= single_fetch_limit:
            total_rows = request.estimated_rows
        else:
            total_rows = self.count_rows_up_to(statement, single_fetch_limit + 1)
        if total_rows > single_fetch_limit:
            total_rows = self.count_rows(statement)
        if request.limit is not None:
            total_rows = min(total_rows, request.limit)

        if total_rows <= 0:
            return SqlPartitionPlan(
                total_rows=0,
                strategy=request.partition_strategy,
                partitions=(),
                meta_dtypes=meta_dtypes,
            )

        if total_rows <= single_fetch_limit:
            # Return a single partition — call offset planner so it
            # compiles the full-statement spec (same as its own ≤chunk_size path).
            partitions = self.plan_offset_partitions(
                statement=statement,
                model=request.model,
                order_column=request.order_column,
                total_rows=total_rows,
                chunk_size=total_rows,
            )
            return SqlPartitionPlan(
                total_rows=total_rows,
                strategy=request.partition_strategy,
                partitions=partitions,
                meta_dtypes=meta_dtypes,
            )

        if request.partition_strategy == "range":
            partitions = self.plan_range_partitions(
                statement=statement,
                model=request.model,
                partition_column=request.partition_column or "",
                total_rows=total_rows,
                chunk_size=request.chunk_size,
            )
        else:
            partitions = self.plan_offset_partitions(
                statement=statement,
                model=request.model,
                order_column=request.order_column,
                total_rows=total_rows,
                chunk_size=request.chunk_size,
            )

        return SqlPartitionPlan(
            total_rows=total_rows,
            strategy=request.partition_strategy,
            partitions=partitions,
            meta_dtypes=meta_dtypes,
        )

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

    def count_rows(self, statement: Select[Any]) -> int:
        count_statement = select(func.count()).select_from(
            self.base_statement(statement).subquery()
        )
        with self.resource.engine.connect() as conn:
            return int(conn.execute(count_statement).scalar_one())

    def count_rows_up_to(self, statement: Select[Any], max_rows: int) -> int:
        limited_statement = self.base_statement(statement).limit(max_rows + 1)
        count_statement = select(func.count()).select_from(limited_statement.subquery())
        with self.resource.engine.connect() as conn:
            return int(conn.execute(count_statement).scalar_one())

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
            total_rows = request.estimated_rows
            if request.limit is not None:
                total_rows = min(total_rows, request.limit)
            if total_rows <= 0:
                return SqlPartitionPlan(
                    total_rows=0,
                    strategy=request.partition_strategy,
                    partitions=(),
                    meta_dtypes=meta_dtypes,
                )
            base = self.base_statement(statement)
            if request.partition_strategy == "offset":
                base = base.limit(request.chunk_size).offset(0)
            return SqlPartitionPlan(
                total_rows=total_rows,
                strategy=request.partition_strategy,
                partitions=(self.compile_partition(base),),
                meta_dtypes=meta_dtypes,
            )

        if request.partition_strategy == "range":
            return await self._async_plan_range(request, statement, meta_dtypes, engine)
        return await self._async_plan_offset(request, statement, meta_dtypes, engine)

    async def _async_plan_range(
        self,
        request: SqlPartitionedLoadRequest,
        statement: Select[Any],
        meta_dtypes: dict[str, str],
        engine: Any,
    ) -> SqlPartitionPlan:
        partitioning_column = _resolve_model_column(request.model, request.partition_column or "")
        async with engine.connect() as count_conn, engine.connect() as bounds_conn:
            total_rows, (lower_bound, upper_bound) = await asyncio.gather(
                self._async_count_rows(statement, count_conn),
                self._async_get_filtered_bounds(statement, partitioning_column, bounds_conn),
            )
        if request.limit is not None:
            total_rows = min(total_rows, request.limit)
        if total_rows <= 0 or lower_bound is None or upper_bound is None:
            return SqlPartitionPlan(
                total_rows=max(0, total_rows),
                strategy=request.partition_strategy,
                partitions=(),
                meta_dtypes=meta_dtypes,
            )
        target_partitions = max(1, math.ceil(total_rows / request.chunk_size))
        target_partitions = min(target_partitions, total_rows)
        bounds = self.build_range_bounds(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target_partitions=target_partitions,
        )
        base_statement = self.base_statement(statement)
        parts: list[SqlPartitionSpec] = []
        for lower, upper in bounds:
            bounded = base_statement.where(partitioning_column >= lower)
            if upper is not None:
                bounded = bounded.where(partitioning_column < upper)
            parts.append(self.compile_partition(bounded))
        partitions: tuple[SqlPartitionSpec, ...] = tuple(parts)
        return SqlPartitionPlan(
            total_rows=total_rows,
            strategy=request.partition_strategy,
            partitions=partitions,
            meta_dtypes=meta_dtypes,
        )

    async def _async_plan_offset(
        self,
        request: SqlPartitionedLoadRequest,
        statement: Select[Any],
        meta_dtypes: dict[str, str],
        engine: Any,
    ) -> SqlPartitionPlan:
        async with engine.connect() as conn:
            total_rows = await self._async_count_rows_up_to(
                statement,
                request.chunk_size + 1,
                conn,
            )
        if total_rows > request.chunk_size:
            async with engine.connect() as conn:
                total_rows = await self._async_count_rows(statement, conn)
        if request.limit is not None:
            total_rows = min(total_rows, request.limit)
        if total_rows <= 0:
            return SqlPartitionPlan(
                total_rows=0,
                strategy=request.partition_strategy,
                partitions=(),
                meta_dtypes=meta_dtypes,
            )

        order_column_name = request.order_column or _infer_primary_key_name(request.model)
        ordering_column = _resolve_model_column(request.model, order_column_name)
        base_statement = self.base_statement(statement).order_by(None).order_by(ordering_column)

        if self._is_range_compatible_column(ordering_column):
            lower_bound, upper_bound = await self._async_compute_ordering_bounds(
                base_statement,
                ordering_column,
                engine,
            )
            if lower_bound is not None and upper_bound is not None:
                partitions = self._plan_keyset_partitions(
                    base_statement=base_statement,
                    ordering_column=ordering_column,
                    total_rows=total_rows,
                    chunk_size=request.chunk_size,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
                return SqlPartitionPlan(
                    total_rows=total_rows,
                    strategy=request.partition_strategy,
                    partitions=partitions,
                    meta_dtypes=meta_dtypes,
                )

        partitions = self.plan_offset_partitions(
            statement=statement,
            model=request.model,
            order_column=request.order_column,
            total_rows=total_rows,
            chunk_size=request.chunk_size,
        )
        return SqlPartitionPlan(
            total_rows=total_rows,
            strategy=request.partition_strategy,
            partitions=partitions,
            meta_dtypes=meta_dtypes,
        )

    async def _async_compute_ordering_bounds(
        self,
        base_statement: Select[Any],
        ordering_column: Any,
        engine: Any,
    ) -> tuple[Any, Any]:
        projection = base_statement.with_only_columns(
            ordering_column,
            maintain_column_froms=True,
        )
        subquery = projection.subquery()
        subquery_column = next(iter(subquery.c))
        bounds_statement = select(func.min(subquery_column), func.max(subquery_column))
        async with engine.connect() as conn:
            result = await conn.execute(bounds_statement)
            return result.one()

    async def _async_count_rows(self, statement: Select[Any], conn: Any) -> int:
        count_statement = select(func.count()).select_from(
            self.base_statement(statement).subquery()
        )
        result = await conn.execute(count_statement)
        return int(result.scalar_one())

    async def _async_count_rows_up_to(
        self, statement: Select[Any], max_rows: int, conn: Any
    ) -> int:
        limited_statement = self.base_statement(statement).limit(max_rows + 1)
        count_statement = select(func.count()).select_from(limited_statement.subquery())
        result = await conn.execute(count_statement)
        return int(result.scalar_one())

    async def _async_get_filtered_bounds(
        self,
        statement: Select[Any],
        partitioning_column: Any,
        conn: Any,
    ) -> tuple[Any, Any]:
        projection = self.base_statement(statement).with_only_columns(
            partitioning_column,
            maintain_column_froms=True,
        )
        subquery = projection.subquery()
        subquery_column = next(iter(subquery.c))
        bounds_statement = select(func.min(subquery_column), func.max(subquery_column))
        result = await conn.execute(bounds_statement)
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
    ) -> tuple[SqlPartitionSpec, ...]:
        order_column_name = order_column or _infer_primary_key_name(model)
        ordering_column = _resolve_model_column(model, order_column_name)
        base_statement = self.base_statement(statement).order_by(None).order_by(ordering_column)

        if self._is_range_compatible_column(ordering_column):
            lower_bound, upper_bound = self._compute_ordering_bounds(
                base_statement,
                ordering_column,
            )
            if lower_bound is not None and upper_bound is not None:
                return self._plan_keyset_partitions(
                    base_statement=base_statement,
                    ordering_column=ordering_column,
                    total_rows=total_rows,
                    chunk_size=chunk_size,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )

        return self._plan_offset_partitions_legacy(
            base_statement=base_statement,
            total_rows=total_rows,
            chunk_size=chunk_size,
        )

    @staticmethod
    def _is_range_compatible_column(column: Any) -> bool:
        return isinstance(column.type, (SmallInteger, Integer, BigInteger))

    def _compute_ordering_bounds(
        self,
        base_statement: Select[Any],
        ordering_column: Any,
    ) -> tuple[Any, Any]:
        projection = base_statement.with_only_columns(
            ordering_column,
            maintain_column_froms=True,
        )
        subquery = projection.subquery()
        subquery_column = next(iter(subquery.c))
        bounds_statement = select(func.min(subquery_column), func.max(subquery_column))
        with self.resource.engine.connect() as conn:
            return conn.execute(bounds_statement).one()

    def _plan_keyset_partitions(
        self,
        *,
        base_statement: Select[Any],
        ordering_column: Any,
        total_rows: int,
        chunk_size: int,
        lower_bound: Any,
        upper_bound: Any,
    ) -> tuple[SqlPartitionSpec, ...]:
        if lower_bound is None or upper_bound is None:
            return (self.compile_partition(base_statement.limit(chunk_size)),)
        if lower_bound == upper_bound:
            return (
                self.compile_partition(
                    base_statement.where(ordering_column >= lower_bound),
                ),
            )

        target_partitions = max(1, math.ceil(total_rows / chunk_size))
        target_partitions = min(target_partitions, total_rows)

        bounds = self.build_numeric_range_bounds(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target_partitions=target_partitions,
        )

        partitions: list[SqlPartitionSpec] = []
        for lower, upper in bounds:
            stmt = base_statement.where(ordering_column >= lower)
            if upper is not None:
                stmt = stmt.where(ordering_column < upper)
            partitions.append(self.compile_partition(stmt))

        return tuple(partitions)

    def _plan_offset_partitions_legacy(
        self,
        *,
        base_statement: Select[Any],
        total_rows: int,
        chunk_size: int,
    ) -> tuple[SqlPartitionSpec, ...]:
        partitions: list[SqlPartitionSpec] = []
        for offset in range(0, total_rows, chunk_size):
            partition_limit = min(chunk_size, total_rows - offset)
            partitions.append(
                self.compile_partition(base_statement.limit(partition_limit).offset(offset))
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
        bounds = self.build_range_bounds(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target_partitions=target_partitions,
        )

        base_statement = self.base_statement(statement)
        partitions: list[SqlPartitionSpec] = []
        for lower, upper in bounds:
            bounded_statement = base_statement.where(partitioning_column >= lower)
            if upper is not None:
                bounded_statement = bounded_statement.where(partitioning_column < upper)
            partitions.append(self.compile_partition(bounded_statement))
        return tuple(partitions)

    def get_filtered_bounds(
        self,
        statement: Select[Any],
        partitioning_column: Any,
    ) -> tuple[Any, Any]:
        projection = self.base_statement(statement).with_only_columns(
            partitioning_column,
            maintain_column_froms=True,
        )
        subquery = projection.subquery()
        subquery_column = next(iter(subquery.c))
        bounds_statement = select(func.min(subquery_column), func.max(subquery_column))
        with self.resource.engine.connect() as conn:
            minimum, maximum = conn.execute(bounds_statement).one()
        return minimum, maximum

    @staticmethod
    def base_statement(statement: Select[Any]) -> Select[Any]:
        return statement.order_by(None).limit(None).offset(None)

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

    @staticmethod
    def infer_meta_dtypes(statement: Select[Any]) -> dict[str, str]:
        meta_dtypes: dict[str, str] = {}
        for selected in statement.selected_columns:
            key = getattr(selected, "key", None) or getattr(selected, "name", None)
            if key is None:
                raise ValueError("partitioned SQL statements must expose named result columns.")
            key_str = str(key)
            if key_str in meta_dtypes:
                raise ValueError(
                    f"partitioned SQL statements must expose unique result column names; duplicate '{key_str}' found."
                )
            meta_dtypes[key_str] = _sqlalchemy_type_to_pandas_dtype(getattr(selected, "type", None))

        if not meta_dtypes:
            raise ValueError("partitioned SQL statements must select at least one result column.")
        return meta_dtypes

    @staticmethod
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
            return SqlPartitionPlanner.build_numeric_range_bounds(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                target_partitions=target_partitions,
            )

        if isinstance(lower_bound, (dt.date, dt.datetime)) and isinstance(
            upper_bound,
            (dt.date, dt.datetime),
        ):
            return SqlPartitionPlanner.build_temporal_range_bounds(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                target_partitions=target_partitions,
            )

        raise ValueError(
            "range partitioning currently supports numeric and temporal partition columns only."
        )

    @staticmethod
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

    @staticmethod
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
                SqlPartitionPlanner.restore_temporal_bound(current, lower_bound),
                SqlPartitionPlanner.restore_temporal_bound(next_value, lower_bound),
            )
            idx += 1
            current = next_value
        return partitions[:idx]

    @staticmethod
    def restore_temporal_bound(value: pd.Timestamp, template: Any) -> Any:
        if isinstance(template, dt.datetime):
            return value.to_pydatetime()
        if isinstance(template, dt.date):
            return value.date()
        return value
