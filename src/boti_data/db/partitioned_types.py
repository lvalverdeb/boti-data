from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import Column
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import ColumnProperty
from sqlalchemy.sql import Select
from sqlalchemy.sql.sqltypes import BigInteger, Integer, SmallInteger

from boti_data.db.partitioned_statements import base_statement

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
        raise AttributeError(f"Column '{column_name}' not found on model '{model.__name__}'")
    return column


def _is_range_compatible_column(column: Any) -> bool:
    return isinstance(column.type, (SmallInteger, Integer, BigInteger))


@dataclass(frozen=True)
class PlannerEngineAdapter:
    """Duck-typed engine/logger/debug shim satisfying SqlPartitionPlanner's resource contract."""

    engine: Any
    logger: Any
    debug: bool = False


@dataclass(frozen=True)
class SqlOffsetPlanContext:
    base_statement: Select[Any]
    ordering_column: Any
    total_rows: int
    chunk_size: int
    limit: int | None
    keyset_eligible: bool


def _prepare_offset_plan_context(
    *,
    statement: Select[Any],
    model: Any,
    order_column: str | None,
    total_rows: int,
    chunk_size: int,
    limit: int | None,
) -> SqlOffsetPlanContext:
    order_column_name = order_column or _infer_primary_key_name(model)
    ordering_column = _resolve_model_column(model, order_column_name)
    ordered_base_statement = base_statement(statement).order_by(None).order_by(ordering_column)
    keyset_eligible = limit is None and _is_range_compatible_column(ordering_column)
    return SqlOffsetPlanContext(
        base_statement=ordered_base_statement,
        ordering_column=ordering_column,
        total_rows=total_rows,
        chunk_size=chunk_size,
        limit=limit,
        keyset_eligible=keyset_eligible,
    )


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
    partition_column: str | None = Field(default=None, max_length=256)
    order_column: str | None = Field(default=None, max_length=256)
    chunk_size: int = Field(default=50_000, ge=1)
    single_fetch_threshold: int | None = Field(default=None, ge=0)
    limit: int | None = Field(default=None, ge=1)
    max_concurrent_fetches: int = Field(
        default=4,
        ge=1,
        le=MAX_PARTITION_FETCH_CONCURRENCY,
    )
    diagnostics: bool = False
    as_pandas: bool = False
    use_arrow: bool = True
    estimated_rows: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_request(self) -> SqlPartitionedLoadRequest:
        self._validate_partitioned_flag()
        self._validate_statement_type()
        self._validate_no_clauses_in_statement()
        self._validate_strategy_config()
        return self

    def _validate_partitioned_flag(self) -> None:
        if self.partitioned is not True:
            raise ValueError("partitioned SQL requests must set partitioned=True.")

    def _validate_statement_type(self) -> None:
        if not isinstance(self.statement, Select):
            raise ValueError("statement must be a SQLAlchemy Select object.")

    def _validate_no_clauses_in_statement(self) -> None:
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

    def _validate_strategy_config(self) -> None:
        if self.partition_strategy == "range":
            self._validate_range_strategy()
        else:
            self._validate_offset_strategy()

    def _validate_range_strategy(self) -> None:
        if not self.partition_column:
            raise ValueError("partition_strategy='range' requires partition_column.")
        _resolve_model_column(self.model, self.partition_column)
        if self.limit is not None:
            raise ValueError(
                "range partitioning does not support request limit; use offset partitioning instead."
            )

    def _validate_offset_strategy(self) -> None:
        order_column = self.order_column or _infer_primary_key_name(self.model)
        _resolve_model_column(self.model, order_column)
