from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import Column
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import ColumnProperty
from sqlalchemy.sql import Select

SqlPartitionParams = tuple[Any, ...] | dict[str, Any] | None
PartitionStrategy = Literal["offset", "range"]
MAX_PARTITION_FETCH_CONCURRENCY = 16


def _infer_primary_key_name(model: Any) -> str:
    mapper = sa_inspect(model)
    primary_keys = [column.key for column in mapper.primary_key]
    if len(primary_keys) != 1:
        raise ValueError(
            "offset partitioning requires order_column unless the model has exactly one primary key."
        )
    return primary_keys[0]


def _resolve_model_column(model: Any, column_name: str) -> Any:
    mapper = sa_inspect(model)
    try:
        mapped_attr = mapper.attrs[column_name]
    except KeyError as exc:
        raise AttributeError(
            f"Column '{column_name}' not found on model '{model.__name__}'"
        ) from exc

    if not isinstance(mapped_attr, ColumnProperty) or len(mapped_attr.columns) != 1:
        raise AttributeError(
            f"Column '{column_name}' is not a directly mapped SQL column on model '{model.__name__}'"
        )

    if not isinstance(mapped_attr.columns[0], Column):
        raise AttributeError(
            f"Column '{column_name}' is not backed by a concrete SQL column on model '{model.__name__}'"
        )

    column = getattr(model, column_name, None)
    if column is None:
        raise AttributeError(
            f"Column '{column_name}' not found on model '{model.__name__}'"
        )
    return column


@dataclass(frozen=True)
class SqlPartitionSpec:
    sql: str
    params: SqlPartitionParams = None


@dataclass(frozen=True)
class SqlPartitionPlan:
    total_rows: int
    strategy: PartitionStrategy
    partitions: tuple[SqlPartitionSpec, ...]
    meta_dtypes: dict[str, str]


class SqlPartitionedLoadRequest(BaseModel):
    """Validated request model for lazy, partitioned SQL -> Dask loading."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    statement: Any
    model: Any
    filters: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    partitioned: bool = True
    partition_strategy: PartitionStrategy = "offset"
    partition_column: str | None = None
    order_column: str | None = None
    chunk_size: int = Field(default=50_000, ge=1)
    limit: int | None = Field(default=None, ge=1)
    max_concurrent_fetches: int = Field(
        default=4,
        ge=1,
        le=MAX_PARTITION_FETCH_CONCURRENCY,
    )
    diagnostics: bool = False
    as_pandas: bool = False
    use_arrow: bool = True

    @model_validator(mode="after")
    def validate_request(self) -> SqlPartitionedLoadRequest:
        if self.partitioned is not True:
            raise ValueError("partitioned SQL requests must set partitioned=True.")
        if not isinstance(self.statement, Select):
            raise ValueError("statement must be a SQLAlchemy Select object.")

        if getattr(self.statement, "_limit_clause", None) is not None:
            raise ValueError(
                "partitioned SQL statements must not include LIMIT; use request limit instead."
            )
        if getattr(self.statement, "_offset_clause", None) is not None:
            raise ValueError(
                "partitioned SQL statements must not include OFFSET; the loader manages partition offsets."
            )
        if getattr(self.statement, "_order_by_clauses", ()):
            raise ValueError(
                "partitioned SQL statements must not include ORDER BY; use order_column instead."
            )

        if self.partition_strategy == "range":
            if not self.partition_column:
                raise ValueError("partition_strategy='range' requires partition_column.")
            _resolve_model_column(self.model, self.partition_column)
            if self.limit is not None:
                raise ValueError(
                    "range partitioning does not support request limit; use offset partitioning instead."
                )
        else:
            order_column = self.order_column or _infer_primary_key_name(self.model)
            _resolve_model_column(self.model, order_column)

        return self
