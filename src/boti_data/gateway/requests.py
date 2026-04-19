from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from boti_data.datacube import DatacubeConfig, DatacubeResource
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.db.sql_resource import SqlDatabaseResource
from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource

from .sql_guard import validate_raw_sql_statement

BackendName = Literal["sqlalchemy", "parquet", "datacube"]
BackendConfig = Union[SqlDatabaseConfig, ParquetDataConfig, DatacubeConfig]
BackendResource = Union[SqlDatabaseResource, ParquetDataResource, DatacubeResource]
ResolvedReturnType = Literal["pandas", "arrow", "dask", "polars"]
ReturnType = Literal["pandas", "arrow", "dask", "polars", "auto"]
ResolvedExecutionMode = Literal["eager", "lazy"]
ExecutionMode = Literal["eager", "lazy", "auto"]


class DataFrameParams(BaseModel):
    """Controls column selection, load behaviour, and output shaping for configured-mode loads.

    Attributes:
        fieldnames: Semantic column names to SELECT from the table.  When set,
            only these columns are fetched (translated to DB column names before
            the query is built).  ``None`` means fetch all columns.
        column_names: Final positional column rename applied *after* the
            field_map semantic rename.  Length must match the number of columns
            in the result.  ``None`` means no override.
        chunk_size: Partition/chunk size passed to the SQL partitioned loader.
            Overrides the runtime ``chunk_size`` kwarg when set here.
        index_col: Column name (semantic) to promote as the DataFrame index
            after load.  Applied after field_map rename.
        datetime_index: Column name (semantic) to parse as datetime and set
            as the DataFrame index.  Takes precedence over ``index_col`` when
            both are set.
        return_type: Output format — ``'dask'`` (default), ``'arrow'`` for
            PyArrow Table, ``'pandas'`` for an eager pandas DataFrame,
            ``'polars'`` for an eager Polars DataFrame, or ``'auto'`` as
            legacy shorthand for pandas-on-small / Dask-on-large behaviour.
        execution_mode: Fetch strategy — ``'lazy'`` for partitioned/Dask-first
            execution, ``'eager'`` for direct eager reads, or ``'auto'`` to
            choose a sensible fetch path from the requested result type.
    """

    model_config = ConfigDict(extra="forbid")

    fieldnames: tuple[str, ...] | None = None
    column_names: list[str] | None = None
    chunk_size: int | None = Field(default=None, ge=1)
    index_col: str | None = None
    datetime_index: str | None = None
    return_type: ReturnType = "dask"
    execution_mode: ExecutionMode = "auto"


class DataFrameOptions(BaseModel):
    """Post-load DataFrame transformations applied in configured-mode gateway loads.

    Operations are applied in order: sort → dedup → groupby.

    Attributes:
        sort_field: Column name (semantic) to sort the result by.
        duplicate_expr: Column(s) used as the subset for ``drop_duplicates``.
            ``None`` means use all columns.
        duplicate_keep: Which duplicate to keep — ``'first'``, ``'last'``,
            or ``False`` to drop all duplicates.
        group_by_expr: Column(s) to group by.  Requires ``group_expr``.
        group_expr: Aggregation mapping passed to ``.agg()``.
            e.g. ``{"amount": "sum", "count": "first"}``.
    """

    model_config = ConfigDict(extra="forbid")

    sort_field: str | None = None
    duplicate_expr: str | list[str] | None = None
    duplicate_keep: Literal["first", "last"] | bool = "last"
    group_by_expr: str | list[str] | None = None
    group_expr: str | dict[str, str] | None = None


class SqlLoadRequest(BaseModel):
    """Validated eager load request for SQL-backed gateway usage."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    sql: str | None = None
    statement: Any | None = None
    model: Any | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    limit: int | None = Field(default=None, ge=0)
    columns: list[str] | None = None
    as_pandas: bool = False
    diagnostics: bool = False
    return_type: ResolvedReturnType = "pandas"
    allow_raw_sql: bool = False

    @model_validator(mode="after")
    def validate_request(self) -> SqlLoadRequest:
        if not self.sql and self.statement is None:
            raise ValueError("Either sql or statement must be provided.")
        if self.sql and self.statement is not None:
            raise ValueError("Provide either sql or statement, not both.")
        if self.sql:
            validate_raw_sql_statement(sql=self.sql, allow_raw_sql=self.allow_raw_sql)
        if self.columns:
            if self.statement is None:
                raise ValueError("columns require a SQLAlchemy statement input.")
            if self.model is None:
                raise ValueError("columns require model for SQLAlchemy column resolution.")
        if self.filters:
            if self.statement is None:
                raise ValueError("filters require a SQLAlchemy statement input.")
            if self.model is None:
                raise ValueError("filters require model for SQLAlchemy column resolution.")
        if self.return_type == "dask":
            # Dask return type requires statement+model (lazy path)
            if self.statement is None or self.model is None:
                raise ValueError(
                    "return_type='dask' requires statement and model for lazy SQL path."
                )
        elif self.return_type == "pandas" and not self.as_pandas:
            raise ValueError(
                "SqlLoadRequest eager return types require as_pandas=True. Use DataGateway.load(...) "
                "with statement and model for the default lazy SQL path, or set as_pandas=True for eager convenience reads."
            )
        return self


class ParquetLoadRequest(BaseModel):
    """Validated load request for parquet-backed gateway usage."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    filters: dict[str, Any] = Field(default_factory=dict)
    raw_filters: list[Any] | None = None
    limit: int | None = Field(default=None, ge=0)
    columns: list[str] | None = None
    as_pandas: bool = False
    diagnostics: bool = False
    return_type: ResolvedReturnType = "pandas"

    @model_validator(mode="after")
    def validate_request(self) -> ParquetLoadRequest:
        if self.filters and self.raw_filters:
            raise ValueError("Use either filters or raw_filters, not both.")
        if self.limit is not None and not self.as_pandas and self.return_type != "arrow":
            raise ValueError(
                "Lazy parquet gateway loads do not support exact limit without triggering eager compute. "
                "Use as_pandas=True, return_type='arrow', or apply Dask head()/partitions explicitly."
            )
        return self


class DatacubeLoadRequest(BaseModel):
    """Validated load request for datacube-backed gateway usage."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    cube: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    limit: int | None = Field(default=None, ge=0)
    columns: list[str] | None = None
    diagnostics: bool = False
    return_type: ResolvedReturnType = "pandas"

