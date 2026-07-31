"""Configured-mode async SQL execution helpers for SqlAlchemyStrategy.

Split out of sql_strategy.py purely for line-count headroom. These take
``ctx``/``resource`` explicitly rather than being methods, since
SqlAlchemyStrategy is stateless and none of this needs ``self``.

``AsyncSqlDatabaseResource`` is referenced through the ``_backend_strategies``
module object (not a direct name import) for the same monkeypatch-compat
reason documented in sql_strategy.py.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd

from . import _backend_strategies
from ._backend_strategies import ConfiguredLoadContext
from .frame_strategies import FrameResult
from .loaders import build_sql_partitioned_request, read_sql_async
from .normalization import build_partitioned_load_options
from .requests import SqlLoadRequest
from .sql_partitioned_exec import run_async_partitioned_request
from .sql_size_estimation import _DEFAULT_PARTITION_CHUNK_SIZE


def trusted_sql_request(
    ctx: ConfiguredLoadContext, *, statement: Any, model: Any
) -> SqlLoadRequest:
    return SqlLoadRequest.model_construct(
        sql=None,
        statement=statement,
        model=model,
        filters=ctx.db_filters or {},
        params=ctx.control.params or {},
        limit=ctx.control.limit,
        columns=None,
        as_pandas=True,
        diagnostics=ctx.control.diagnostics,
        return_type=ctx.loader_return_type,
    )


async def aload_configured_execute_async_resource(
    ctx: ConfiguredLoadContext, resource: Any
) -> FrameResult:
    diagnostics = ctx.control.diagnostics
    assert ctx.get_configured_select_async is not None
    t0 = perf_counter()
    model, stmt = await ctx.get_configured_select_async(resource, ctx.db_columns)
    t1 = perf_counter()
    if diagnostics:
        resource.logger.info(f"Configured async select reflect_select={t1 - t0:.3f}s")  # nosec CWE-117 -- float only
    if ctx.execution_mode == "lazy":
        t2 = perf_counter()
        partitioned_options = build_partitioned_load_options(
            statement=stmt,
            model=model,
            filters=ctx.db_filters,
            control=ctx.control.model_dump(),
            default_chunk_size=(ctx.chunk_size or _DEFAULT_PARTITION_CHUNK_SIZE),
        )
        options_dict = partitioned_options.model_dump(exclude_none=True)
        options_dict["use_arrow"] = False
        request = build_sql_partitioned_request(options_dict)
        t3 = perf_counter()
        if diagnostics:
            resource.logger.info(f"Build partitioned request elapsed={t3 - t2:.3f}s")  # nosec CWE-117 -- float only
        return await run_async_partitioned_request(ctx, resource, request)

    df = await read_sql_async(resource, trusted_sql_request(ctx, statement=stmt, model=model))
    if isinstance(df, pd.DataFrame):
        df = ctx.post_processor.coerce_eager_sql_frame(df, statement=stmt)
    return df


async def aload_configured_async_sql(ctx: ConfiguredLoadContext) -> FrameResult:
    assert ctx.get_configured_select_async is not None

    async def _execute(resource: Any) -> FrameResult:
        return await aload_configured_execute_async_resource(ctx, resource)

    async_resource = ctx.async_sql_resource
    if async_resource is None:
        async with _backend_strategies.AsyncSqlDatabaseResource(ctx.config) as resource:
            df = await _execute(resource)
    else:
        df = await _execute(async_resource)

    return ctx.post_processor.finalize_configured_result(
        df,
        return_type=ctx.return_type,
        apply_field_map=True,
        fieldnames=ctx.configured_fieldnames,
    )
