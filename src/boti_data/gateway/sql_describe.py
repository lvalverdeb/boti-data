"""Cheap SQL table introspection backing DataGateway.describe()/adescribe().

Split out of sql_strategy.py purely for line-count headroom, mirroring the
sql_size_estimation.py module split.
"""

from __future__ import annotations

from typing import Any

from boti_data.db.partitioned_planner import SqlPartitionPlanner
from boti_data.db.partitioned_statements import infer_meta_dtypes

from .loaders import reflect_and_select
from .requests import TableDescription


def describe_table(
    resource: Any,
    table: str,
    *,
    row_count_limit: int,
) -> TableDescription:
    """Reflect *table*'s columns and count rows up to *row_count_limit*.

    Never loads any actual row data: column dtypes come from SQLAlchemy
    reflection (no query executed), and the row count is a single bounded
    ``COUNT(*)`` subquery capped at *row_count_limit* rows — safe to run
    against an unfamiliar table without risking an accidental unbounded
    query against a large production table.
    """
    _model, statement = reflect_and_select(resource, table)
    columns = infer_meta_dtypes(statement)
    counted = SqlPartitionPlanner(resource).count_rows_up_to(statement, row_count_limit)
    is_exact = counted <= row_count_limit
    return TableDescription(
        table=table,
        columns=columns,
        row_count=counted if is_exact else row_count_limit,
        row_count_is_exact=is_exact,
    )
