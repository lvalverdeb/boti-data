from __future__ import annotations

from typing import Any

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from sqlalchemy import select, text

from boti_data.db import (
    SqlDatabaseConfig,
    SqlDatabaseResource,
    SqlPartitionedLoader,
    SqlPartitionedLoadRequest,
)
from boti_data.db.arrow_schema_mapper import (
    arrow_table_to_pandas,
    build_arrow_schema_from_sqlalchemy_types,
    rows_to_arrow_table,
)
from boti_data.db.sql_model_builder import SqlAlchemyModelBuilder
from boti_data.filters import FilterHandler
from boti_data.parquet.resource import ParquetDataResource

from .requests import (
    ParquetLoadRequest,
    SqlLoadRequest,
)


def _prepare_sql_statement(
    request: SqlLoadRequest, *, logger: Any, debug: bool
) -> tuple[Any, dict[str, Any] | None]:
    if request.statement is not None:
        statement = request.statement
        if request.columns:
            projected_columns = [getattr(request.model, column) for column in request.columns]
            statement = statement.with_only_columns(*projected_columns, maintain_column_froms=True)
        if request.params:
            statement = statement.params(**request.params)
        execute_params = None
    else:
        statement = text(request.sql or "")
        execute_params = request.params or None

    if request.filters:
        handler = FilterHandler(
            backend="sqlalchemy",
            logger=logger,
            debug=debug,
        )
        statement = handler.apply_filters(statement, model=request.model, filters=request.filters)
    if request.limit is not None and hasattr(statement, "limit"):
        statement = statement.limit(request.limit)
    return statement, execute_params


def _arrow_table_from_sql_result(
    rows: list[tuple[Any, ...]],
    columns: list[str],
    *,
    statement: Any | None,
) -> pa.Table:
    if statement is None:
        if not rows:
            return pa.table({name: [] for name in columns})
        return pa.Table.from_pylist([dict(zip(columns, row)) for row in rows])

    sql_types = [getattr(selected, "type", None) for selected in statement.selected_columns]
    schema = build_arrow_schema_from_sqlalchemy_types(columns, sql_types)
    return rows_to_arrow_table(rows, columns, schema)


def should_use_partitioned_sql(options: dict[str, Any]) -> bool:
    partitioned = options.get("partitioned")
    as_pandas = bool(options.get("as_pandas", False))
    if partitioned is False:
        if not as_pandas:
            raise ValueError(
                "Non-partitioned SQL gateway loads are pandas-only. "
                "Set as_pandas=True, or omit partitioned to use the default lazy path."
            )
        return False
    if partitioned is True:
        return True
    return not as_pandas


def build_sql_partitioned_request(options: dict[str, Any]) -> SqlPartitionedLoadRequest:
    partitioned_options = dict(options)

    if partitioned_options.get("sql") is not None:
        raise ValueError(
            "Lazy SQL gateway loads require a SQLAlchemy Select statement and model. "
            "Use as_pandas=True for raw SQL convenience reads."
        )
    if partitioned_options.get("statement") is None:
        raise ValueError(
            "Lazy SQL gateway loads require statement and model. "
            "Use as_pandas=True for eager SQL convenience reads."
        )
    if partitioned_options.get("model") is None:
        raise ValueError(
            "Lazy SQL gateway loads require model for partition planning and SQL column resolution."
        )

    # Discard any extra fields that SqlPartitionedLoadRequest does not accept
    # (e.g. dry_run, resilient — control keys passed through from the gateway).
    partitioned_options = {
        k: v for k, v in partitioned_options.items() if k in SqlPartitionedLoadRequest.model_fields
    }
    partitioned_options.setdefault("as_pandas", False)
    partitioned_options.setdefault("partitioned", True)

    return SqlPartitionedLoadRequest.model_validate(partitioned_options)


def _finalize_arrow_result(
    rows: list[tuple[Any, ...]],
    columns: list[str],
    *,
    statement: Any | None,
    limit: int | None,
) -> pa.Table:
    table = _arrow_table_from_sql_result(rows, columns, statement=statement)
    if limit is not None:
        table = table.slice(0, limit)
    return table


def _finalize_frame_result(frame: pd.DataFrame, *, limit: int | None) -> pd.DataFrame:
    if limit is not None:
        frame = frame.head(limit)
    return frame


def load_sql(
    resource: SqlDatabaseResource,
    request: SqlLoadRequest,
) -> pd.DataFrame | pa.Table:
    statement, execute_params = _prepare_sql_statement(
        request,
        logger=resource.logger,
        debug=resource.debug,
    )

    with resource.engine.connect() as conn:
        if request.return_type == "arrow":
            result = conn.execute(statement, execute_params or {})
            return _finalize_arrow_result(
                [tuple(row) for row in result.fetchall()],
                list(result.keys()),
                statement=statement,
                limit=request.limit,
            )
        frame = pd.read_sql(statement, conn, params=execute_params)

    return _finalize_frame_result(frame, limit=request.limit)


def load_sql_partitioned(
    config: SqlDatabaseConfig,
    resource: SqlDatabaseResource,
    request: SqlPartitionedLoadRequest,
) -> pd.DataFrame | dd.DataFrame:
    # resource is caller-owned (SqlPartitionedLoader._owns_resource is False here),
    # so closing the loader only releases its own planner/executor state and does
    # not touch resource — safe to close deterministically instead of relying on GC.
    with SqlPartitionedLoader(config, resource=resource, use_arrow=request.use_arrow) as loader:
        return loader.load_request(request)


async def read_sql_async(
    resource: Any,
    request: SqlLoadRequest,
) -> pd.DataFrame | pa.Table:
    statement, execute_params = _prepare_sql_statement(
        request,
        logger=resource.logger,
        debug=False,
    )

    async with resource.engine.connect() as conn:
        if request.return_type == "arrow":
            result = await conn.execute(statement, execute_params or {})
            return _finalize_arrow_result(
                [tuple(row) for row in result.fetchall()],
                list(result.keys()),
                statement=statement,
                limit=request.limit,
            )
        frame = await conn.run_sync(
            lambda sync_conn: pd.read_sql(statement, sync_conn, params=execute_params)
        )

    return _finalize_frame_result(frame, limit=request.limit)


def _load_parquet_eager(
    resource: ParquetDataResource,
    request: ParquetLoadRequest,
) -> pa.Table | pd.DataFrame:
    if request.filters:
        table = resource.load_filtered_arrow(request.filters, columns=request.columns)
    else:
        table = resource.load_arrow(filters=request.raw_filters, columns=request.columns)
    if request.limit is not None:
        table = table.slice(0, request.limit)
    if request.as_pandas:
        return arrow_table_to_pandas(table)
    return table


def _load_parquet_lazy(
    resource: ParquetDataResource,
    request: ParquetLoadRequest,
) -> pd.DataFrame | dd.DataFrame:
    if request.filters:
        frame = resource.load_filtered(request.filters, columns=request.columns)
    else:
        frame = resource.load_files(filters=request.raw_filters, columns=request.columns)

    if request.limit is not None:
        limited = frame.head(request.limit, npartitions=-1, compute=True)
        if request.as_pandas:
            return limited
        raise ValueError(
            "Lazy parquet gateway loads do not support exact limit without eager compute. "
            "Use as_pandas=True or apply Dask head()/partitions explicitly."
        )

    if request.as_pandas:
        return frame.compute()
    return frame


def load_parquet(
    resource: ParquetDataResource,
    request: ParquetLoadRequest,
) -> pa.Table | pd.DataFrame | dd.DataFrame:
    if request.return_type == "arrow" or request.as_pandas:
        return _load_parquet_eager(resource, request)
    return _load_parquet_lazy(resource, request)


def _select_from_model(model: Any, db_column_names: list[str] | None) -> Any:
    if db_column_names:
        columns = [getattr(model, col) for col in db_column_names]
        return select(*columns)
    return select(model)


# Not a copy-pasted twin: both already share _select_from_model(); the
# remaining difference is the irreducible await builder.build_model_async()
# vs builder.build_model() call.
# spaghetti-ignore[sync-async-duplication]: see above
async def reflect_and_select_async(
    resource: Any,
    table: str,
    db_column_names: list[str] | None = None,
) -> tuple[Any, Any]:
    """Async variant of :func:`reflect_and_select` for use with async DSNs.

    Args:
        resource: Active :class:`AsyncSqlDatabaseResource`.
        table: DB table name (no schema prefix).
        db_column_names: Optional list of **DB column names** to project.

    Returns:
        ``(model, statement)`` — the reflected ORM class and the
        :class:`sqlalchemy.sql.Select` object ready for async loading.
    """
    builder = SqlAlchemyModelBuilder(resource.engine, table)
    model = await builder.build_model_async()
    return model, _select_from_model(model, db_column_names)


def reflect_and_select(
    resource: SqlDatabaseResource,
    table: str,
    db_column_names: list[str] | None = None,
) -> tuple[Any, Any]:
    """Reflect *table* via the model registry and return ``(model, Select)``.

    Args:
        resource: Active :class:`SqlDatabaseResource`.
        table: DB table name (no schema prefix).
        db_column_names: Optional list of **DB column names** to project.
            When ``None`` the statement selects all columns (``select(model)``).

    Returns:
        ``(model, statement)`` — the reflected ORM class and the
        :class:`sqlalchemy.sql.Select` object ready for partitioned loading.
    """
    builder = SqlAlchemyModelBuilder(resource.engine, table)
    model = builder.build_model()
    return model, _select_from_model(model, db_column_names)
