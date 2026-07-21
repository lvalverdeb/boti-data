"""Auto-return-type result-size estimation for SqlAlchemyStrategy.

Split out of SqlAlchemyStrategy: this whole cluster (structured/configured,
sync/async probing) is only ever called through
``estimate_result_size``/``estimate_result_size_async``, and referenced by
nothing outside the strategy — so it moves here wholesale as plain
functions (the strategy itself holds no state to close over).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from boti_data.db import SqlDatabaseResource, SqlPartitionedLoadRequest
from boti_data.db.partitioned_planner import SqlPartitionPlanner
from boti_data.db.partitioned_types import PlannerEngineAdapter

from .loaders import build_sql_partitioned_request
from .normalization import build_partitioned_load_options

if TYPE_CHECKING:
    from ._backend_strategies import ConfiguredLoadContext, StructuredLoadContext

_DEFAULT_PARTITION_CHUNK_SIZE: int = SqlPartitionedLoadRequest.model_fields["chunk_size"].default

_AUTO_EAGER_MAX_ROWS = 10_000

_NO_CONFIGURED_SHORTCUT = object()


def _auto_sql_threshold(limit: int | None = None) -> int:
    effective_limit = _AUTO_EAGER_MAX_ROWS if limit is None else min(limit, _AUTO_EAGER_MAX_ROWS)
    return max(0, effective_limit)


def early_structured_estimate(
    ctx: StructuredLoadContext,
) -> tuple[SqlPartitionedLoadRequest, int] | tuple[None, None]:
    """Returns (request, threshold) to probe, or (None, None) to abstain."""
    options = ctx.opts
    if (
        options.get("partitioned") is True
        or options.get("sql") is not None
        or options.get("statement") is None
        or options.get("model") is None
    ):
        return None, None
    request = build_sql_partitioned_request(options)
    threshold = _auto_sql_threshold(request.limit)
    if threshold == 0:
        return None, None
    return request, threshold


def probe_sync(
    resource: SqlDatabaseResource,
    request: SqlPartitionedLoadRequest,
    threshold: int,
) -> tuple[int, None]:
    planner = SqlPartitionPlanner(resource)
    statement = planner.prepare_statement(request)
    probed_rows = min(planner.count_rows_up_to(statement, threshold), threshold)
    return probed_rows, None


# Not a copy-pasted twin: estimate_structured_async() has a genuinely
# different decision tree — it must additionally choose between a
# same-thread probe_sync() (when ctx.resource is set) and a fully-async
# probe_async() fallback via ctx.async_sql_resource that this has no
# equivalent for; only the leading early_structured_estimate() call is
# literally shared, too trivial to extract further.
# spaghetti-ignore[sync-async-duplication]: see above
def estimate_structured(
    ctx: StructuredLoadContext,
) -> tuple[int | None, int | None] | None:
    request, threshold = early_structured_estimate(ctx)
    if request is None:
        return None
    if not isinstance(ctx.resource, SqlDatabaseResource):
        return None
    return probe_sync(ctx.resource, request, threshold)


def configured_early_estimate(ctx: ConfiguredLoadContext) -> Any:
    """Returns an estimate tuple/None if resolvable without a probe, else a sentinel."""
    if ctx.control.partitioned is True:
        return None
    limit = ctx.control.limit
    if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
        return limit, None
    return _NO_CONFIGURED_SHORTCUT


def build_configured_probe_request(
    ctx: ConfiguredLoadContext, stmt: Any, model: Any
) -> SqlPartitionedLoadRequest:
    partitioned_options = build_partitioned_load_options(
        statement=stmt,
        model=model,
        filters=ctx.db_filters,
        control=ctx.control.model_dump(),
        default_chunk_size=ctx.chunk_size or _DEFAULT_PARTITION_CHUNK_SIZE,
    )
    return build_sql_partitioned_request(partitioned_options.model_dump(exclude_none=True))


def estimate_configured(
    ctx: ConfiguredLoadContext,
) -> tuple[int | None, int | None] | None:
    shortcut = configured_early_estimate(ctx)
    if shortcut is not _NO_CONFIGURED_SHORTCUT:
        return shortcut
    if not isinstance(ctx.resource, SqlDatabaseResource) or ctx.get_configured_select is None:
        return None
    model, stmt = ctx.get_configured_select(ctx.db_columns)
    request = build_configured_probe_request(ctx, stmt, model)
    threshold = _auto_sql_threshold(request.limit)
    return None if threshold == 0 else probe_sync(ctx.resource, request, threshold)


async def estimate_structured_async(
    ctx: StructuredLoadContext,
) -> tuple[int | None, int | None] | None:
    request, threshold = early_structured_estimate(ctx)
    if request is None:
        return None
    if ctx.resource is not None:
        assert isinstance(ctx.resource, SqlDatabaseResource)
        return probe_sync(ctx.resource, request, threshold)
    return await probe_async(request, threshold, ctx.async_sql_resource)


async def estimate_configured_async(
    ctx: ConfiguredLoadContext,
) -> tuple[int | None, int | None] | None:
    shortcut = configured_early_estimate(ctx)
    if shortcut is not _NO_CONFIGURED_SHORTCUT:
        return shortcut
    assert ctx.get_configured_select_async is not None
    model, stmt = await ctx.get_configured_select_async(
        ctx.async_sql_resource,
        ctx.db_columns,
    )
    request = build_configured_probe_request(ctx, stmt, model)
    return await probe_async(
        request,
        _auto_sql_threshold(request.limit),
        ctx.async_sql_resource,
    )


async def probe_async(
    request: SqlPartitionedLoadRequest,
    threshold: int,
    async_resource: Any,
) -> tuple[int | None, int | None] | None:
    if threshold == 0:
        return None
    if async_resource is None:
        return None
    adapter = PlannerEngineAdapter(
        engine=async_resource.engine.sync_engine,
        logger=async_resource.logger,
        debug=False,
    )
    planner = SqlPartitionPlanner(adapter)  # type: ignore[arg-type]
    statement = planner.prepare_statement(request)
    async with async_resource.engine.connect() as conn:
        probed_rows = await planner._async_count_rows_up_to(statement, threshold, conn)
    return min(probed_rows, threshold), None
