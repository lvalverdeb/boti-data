"""Pure SQLAlchemy Select-shaping helpers for SqlPartitionPlanner."""

from __future__ import annotations

from typing import Any

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


def base_statement(statement: Select[Any]) -> Select[Any]:
    return statement.order_by(None).limit(None).offset(None)


def bounds_statement(base_stmt: Select[Any], ordering_column: Any) -> Select[Any]:
    projection = base_stmt.with_only_columns(
        ordering_column,
        maintain_column_froms=True,
    )
    subquery = projection.subquery()
    subquery_column = next(iter(subquery.c))
    return select(func.min(subquery_column), func.max(subquery_column))


def count_statement(statement: Select[Any]) -> Select[Any]:
    return select(func.count()).select_from(base_statement(statement).subquery())


def count_up_to_statement(statement: Select[Any], max_rows: int) -> Select[Any]:
    limited_statement = base_statement(statement).limit(max_rows + 1)
    return select(func.count()).select_from(limited_statement.subquery())


_SQL_TYPE_TO_PANDAS_DTYPE: list[tuple[tuple[type, ...], str]] = [
    ((SmallInteger, Integer, BigInteger), "Int64"),
    ((Numeric, Float), "Float64"),
    ((Boolean,), "boolean"),
    ((Date, DateTime), "datetime64[ns, UTC]"),
    ((String, Text, Unicode, UnicodeText, Time), "string"),
    ((LargeBinary,), "object"),
]


def _sqlalchemy_type_to_pandas_dtype(sql_type: Any) -> str:
    for types, dtype in _SQL_TYPE_TO_PANDAS_DTYPE:
        if isinstance(sql_type, types):
            return dtype
    return "object"


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
