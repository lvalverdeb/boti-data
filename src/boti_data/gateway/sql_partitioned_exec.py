"""Shared async partitioned/chunked SQL execution engine.

Split out of SqlAlchemyStrategy: this is the biggest single cluster of
strategy internals (the adaptive single-fetch fast path plus full keyset
planning/fetch fallback), called from both structured-mode and
configured-mode async loads, and referenced by nothing outside the
strategy — so it moves here wholesale rather than staying as a thin
wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
import pandas as pd

from boti_data.db import SqlDatabaseConfig, SqlPartitionedLoadRequest
from boti_data.db.partitioned_execution import (
    GateWaitStats,
    SqlPartitionExecutor,
    fetch_gate_stats,
    format_gate_wait_suffix,
)
from boti_data.db.partitioned_planner import SqlPartitionPlanner
from boti_data.db.partitioned_types import PlannerEngineAdapter
from boti_data.db.sql_config import WorkerSqlConfig
from boti_data.db.sql_engine import _get_worker_engine_identity

from .frame_strategies import FrameResult

if TYPE_CHECKING:
    from ._backend_strategies import StructuredLoadContext

# Adaptive single-fetch ceiling for the partitioned-dask fast path.
#
# The ceiling represents "the largest result we'll materialize in one process" —
# i.e. the pandas/polars comfort zone. That is fundamentally a *memory* limit, not
# a fixed row count: a narrow 4-int result and a 40-column string result of the
# same row count have wildly different footprints. So we size the ceiling in BYTES
# and convert to a row count using the estimated width of a result row (from the
# inferred column dtypes). A narrow table therefore single-fetches many more rows
# than a wide one, and the ceiling tracks the actual choke point rather than a
# blind guess.
#
# Below the ceiling we stream the whole result in one scan and return a single
# partition (partitioning a result that fits in memory is pure overhead — a COUNT
# round-trip plus per-partition fetch coordination). Above it we fall through to
# keyset partitioning and return a lazy dask frame. The ceiling also bounds how
# many rows the probe streams before deciding to partition, so it doubles as the
# probe's memory bound.
#
# ``_DEFAULT_SINGLE_FETCH_BYTES`` is the estimated *final-frame* size; peak memory
# during read_sql construction can be ~2-3x, so the budget stays well under
# typical process RAM. A per-call ``single_fetch_threshold`` (an explicit row
# count) overrides the byte-derived ceiling when set.
_DEFAULT_SINGLE_FETCH_BYTES: int = 512 * 1024 * 1024

# Absolute row safety-rail on the byte-derived ceiling: keeps a pathologically
# narrow schema (e.g. a single small column) from yielding a tens-of-millions-row
# single fetch just because the bytes fit — that many rows carries too much
# per-object / construction overhead to treat as one comfortable partition.
_MAX_SINGLE_FETCH_ROWS: int = 5_000_000

# Estimated in-memory bytes per value, keyed by the pandas dtype that
# ``_sqlalchemy_type_to_pandas_dtype`` infers. Variable-width text ("string" /
# "object") is charged a deliberately conservative flat cost so string-heavy rows
# partition earlier rather than risk OOM — over-estimating width is the safe
# direction (it lowers the row ceiling).
_DTYPE_BYTE_WEIGHTS: dict[str, int] = {
    "Int64": 8,
    "Float64": 8,
    "boolean": 1,
    "datetime64[ns, UTC]": 8,
    "string": 64,
    "object": 64,
}
_FALLBACK_DTYPE_BYTES: int = 16


@dataclass(frozen=True)
class _FastPathTiming:
    """perf_counter() checkpoints shared by the fast-path result and its fallback log."""

    started: float
    t_planner_created: float
    t_prepare: float
    t_fetch: float
    t_coerce: float | None = None


def _estimate_bytes_per_row(meta_dtypes: dict[str, str]) -> int:
    """Estimate the in-memory footprint of one result row from its column dtypes."""
    total = sum(
        _DTYPE_BYTE_WEIGHTS.get(dtype, _FALLBACK_DTYPE_BYTES) for dtype in meta_dtypes.values()
    )
    return max(1, total)


def _resolve_single_fetch_ceiling(
    *,
    chunk_size: int,
    meta_dtypes: dict[str, str],
    override_rows: int | None,
) -> int:
    """Row ceiling below which the fast path returns a single partition.

    An explicit ``override_rows`` (the request's ``single_fetch_threshold``) wins;
    otherwise derive it from the byte budget and the estimated row width, capped by
    ``_MAX_SINGLE_FETCH_ROWS`` and floored at ``chunk_size`` (a single chunk always
    fits in one partition).
    """
    if override_rows is not None:
        return max(chunk_size, override_rows)
    budget_rows = _DEFAULT_SINGLE_FETCH_BYTES // _estimate_bytes_per_row(meta_dtypes)
    budget_rows = min(budget_rows, _MAX_SINGLE_FETCH_ROWS)
    return max(chunk_size, budget_rows)


# Adaptive fast path: bypass the planner entirely when the whole result
# fits under the single-fetch ceiling. Stream up to ceiling + 1 rows in a
# single scan; if we get <= ceiling, return one partition immediately. Only
# when the result overflows do we fall through to keyset planning (a COUNT
# round-trip + per-partition fetches). This collapses the small AND medium
# tiers into one cheap scan — partitioning a result that fits in one process
# is pure overhead (benchmarks: a ~1.7M-row single fetch ~7s beat the
# multi-partition path ~15s).
#
# The ceiling is a memory budget converted to rows via the estimated row
# width (see _resolve_single_fetch_ceiling), never below ``chunk_size``, and
# overridable per-call by ``single_fetch_threshold``. It also bounds how many
# rows the probe buffers before deciding to partition, so it doubles as the
# probe's memory bound.
#
# The probe streams via a server-side cursor and stops after the cap rather
# than adding a SQL ``LIMIT``: on MySQL a ``LIMIT`` on a non-covering filtered
# scan flips the optimizer to a ~3x slower index-driven plan (verified against
# asm_tracking_productos — no-LIMIT ~4s vs LIMIT ~12s, unhelped by ORDER BY or a
# derived-table wrapper). Streaming keeps the fast full-scan plan while still
# bounding client memory to the cap. See SqlPartitionExecutor.probe_capped.
def _build_fast_path_result(
    df: pd.DataFrame,
    meta_dtypes: dict[str, str],
    request: SqlPartitionedLoadRequest,
    async_resource: Any,
    *,
    single_fetch_ceiling: int,
    timing: _FastPathTiming,
) -> FrameResult:
    df = SqlPartitionExecutor.align_and_coerce_partition(df, meta_dtypes)
    t_dask = perf_counter()
    result = dd.from_pandas(df, npartitions=1)
    t_end = perf_counter()
    if request.diagnostics:
        async_resource.logger.info(
            "Partitioned SQL fast path: single partition "
            f"rows={len(df)} ceiling={single_fetch_ceiling} "
            f"est_bytes_per_row={_estimate_bytes_per_row(meta_dtypes)} "
            f"chunk_size={request.chunk_size} "
            f"total={t_end - timing.started:.3f}s "
            f"planner_init={timing.t_prepare - timing.t_planner_created:.3f}s "
            f"prepare_stmt={timing.t_fetch - timing.t_prepare:.3f}s "
            f"db_fetch={timing.t_coerce - timing.t_fetch:.3f}s "
            f"coerce={t_dask - timing.t_coerce:.3f}s "
            f"from_pandas={t_end - t_dask:.3f}s"
        )
    return result


def _log_fast_path_fallback(
    async_resource: Any,
    request: SqlPartitionedLoadRequest,
    meta_dtypes: dict[str, str],
    rows_fetched: int,
    single_fetch_ceiling: int,
    timing: _FastPathTiming,
) -> None:
    if not request.diagnostics:
        return
    async_resource.logger.info(
        "Partitioned SQL fast path fallback: rows exceed ceiling, "
        f"rows_fetched={rows_fetched} ceiling={single_fetch_ceiling} "
        f"est_bytes_per_row={_estimate_bytes_per_row(meta_dtypes)} "
        f"chunk_size={request.chunk_size} "
        f"prepare_stmt={timing.t_fetch - timing.t_prepare:.3f}s "
        f"db_fetch={perf_counter() - timing.t_fetch:.3f}s"
    )


async def _try_fast_path(
    async_resource: Any,
    request: SqlPartitionedLoadRequest,
    planner: SqlPartitionPlanner,
    *,
    started: float,
    t_planner_created: float,
) -> FrameResult | None:
    """Return a single-partition result if it fits under the fast-path ceiling, else None."""
    t_prepare = perf_counter()
    prepared_stmt = planner.prepare_statement(request)
    meta_dtypes = planner.infer_meta_dtypes(prepared_stmt)
    single_fetch_ceiling = _resolve_single_fetch_ceiling(
        chunk_size=request.chunk_size,
        meta_dtypes=meta_dtypes,
        override_rows=request.single_fetch_threshold,
    )
    probe_cap = single_fetch_ceiling + 1
    if request.limit is not None:
        probe_cap = min(request.limit, probe_cap)
    t_fetch = perf_counter()
    timing = _FastPathTiming(
        started=started, t_planner_created=t_planner_created, t_prepare=t_prepare, t_fetch=t_fetch
    )
    async with async_resource.engine.connect() as conn:
        df = await conn.run_sync(
            lambda sync_conn: SqlPartitionExecutor.probe_capped(
                sync_conn, prepared_stmt, probe_cap
            ),
        )
        if len(df) <= single_fetch_ceiling:
            timing = replace(timing, t_coerce=perf_counter())
            return _build_fast_path_result(
                df,
                meta_dtypes,
                request,
                async_resource,
                single_fetch_ceiling=single_fetch_ceiling,
                timing=timing,
            )
        _log_fast_path_fallback(
            async_resource, request, meta_dtypes, len(df), single_fetch_ceiling, timing
        )
    return None


def _log_plan_summary(
    async_resource: Any,
    request: SqlPartitionedLoadRequest,
    plan: Any,
    *,
    t_plan: float,
    t_plan_done: float,
) -> None:
    if not request.diagnostics:
        return
    async_resource.logger.info(
        "Partitioned SQL plan "
        f"strategy={plan.strategy} partitions={len(plan.partitions)} "
        f"rows={plan.total_rows} "
        f"chunk_size={request.chunk_size} "
        f"max_concurrent_fetches={request.max_concurrent_fetches} "
        f"use_arrow={request.use_arrow} "
        f"plan_elapsed={t_plan_done - t_plan:.3f}s"
    )


async def _execute_single_partition_plan(
    planner: SqlPartitionPlanner,
    async_resource: Any,
    request: SqlPartitionedLoadRequest,
    plan: Any,
) -> FrameResult:
    t_prepare2 = perf_counter()
    prepared_stmt = planner.prepare_statement(request)
    t_fetch2 = perf_counter()
    async with async_resource.engine.connect() as conn:
        df = await conn.run_sync(
            lambda sync_conn: pd.read_sql(prepared_stmt, sync_conn),
        )
    t_coerce2 = perf_counter()
    df = SqlPartitionExecutor.align_and_coerce_partition(df, plan.meta_dtypes)
    t_dask2 = perf_counter()
    result = dd.from_pandas(df, npartitions=1)
    t_end2 = perf_counter()
    if request.diagnostics:
        async_resource.logger.info(
            "Partitioned SQL single-partition plan path "
            f"rows={len(df)} "
            f"prepare_stmt={t_fetch2 - t_prepare2:.3f}s "
            f"db_fetch={t_coerce2 - t_fetch2:.3f}s "
            f"coerce={t_dask2 - t_coerce2:.3f}s "
            f"from_pandas={t_end2 - t_dask2:.3f}s"
        )
    return result


def _execute_via_partition_executor(
    planner: SqlPartitionPlanner,
    request: SqlPartitionedLoadRequest,
    plan: Any,
    worker_config: WorkerSqlConfig,
    gate_key: str,
) -> FrameResult:
    executor = SqlPartitionExecutor(
        worker_config,
        gate_key,
        use_arrow=request.use_arrow,
    )
    prepared_stmt = planner.prepare_statement(request)
    return executor.load_plan(
        plan,
        as_pandas=request.as_pandas,
        max_concurrent_fetches=request.max_concurrent_fetches,
        fetch_partition=SqlPartitionExecutor.fetch_partition_async,
        statement=prepared_stmt,
    )


def _log_completion(
    async_resource: Any,
    request: SqlPartitionedLoadRequest,
    plan: Any,
    result: FrameResult,
    started: float,
    gate_context: tuple[str, GateWaitStats | None] | None = None,
) -> None:
    if not request.diagnostics:
        return
    result_partitions = result.npartitions if isinstance(result, dd.DataFrame) else 1
    gate_suffix = "" if gate_context is None else format_gate_wait_suffix(*gate_context)
    async_resource.logger.info(
        "Partitioned SQL load completed "
        f"strategy={plan.strategy} partitions={len(plan.partitions)} "
        f"rows={plan.total_rows} result_partitions={result_partitions} "
        f"elapsed={perf_counter() - started:.2f}s"
        f"{gate_suffix}"
    )


async def run_async_partitioned_request(
    ctx: StructuredLoadContext,
    async_resource: Any,
    request: SqlPartitionedLoadRequest,
) -> FrameResult:
    assert isinstance(ctx.config, SqlDatabaseConfig)

    started = perf_counter()
    t_planner_created = perf_counter()
    adapter = PlannerEngineAdapter(
        engine=async_resource.engine.sync_engine,
        logger=async_resource.logger,
        debug=False,
    )
    planner = SqlPartitionPlanner(adapter)  # type: ignore[arg-type]

    if not request.as_pandas:
        fast_result = await _try_fast_path(
            async_resource,
            request,
            planner,
            started=started,
            t_planner_created=t_planner_created,
        )
        if fast_result is not None:
            return fast_result

    t_plan = perf_counter()
    plan = await planner.async_plan_request(request, async_resource.engine)
    _log_plan_summary(async_resource, request, plan, t_plan=t_plan, t_plan_done=perf_counter())

    worker_config = WorkerSqlConfig.from_database_config(ctx.config)
    gate_key = _get_worker_engine_identity(worker_config)

    gate_baseline = fetch_gate_stats().get(gate_key) if request.diagnostics else None
    if len(plan.partitions) == 1 and not request.as_pandas:
        result = await _execute_single_partition_plan(planner, async_resource, request, plan)
    else:
        result = _execute_via_partition_executor(planner, request, plan, worker_config, gate_key)

    _log_completion(async_resource, request, plan, result, started, (gate_key, gate_baseline))
    return result
