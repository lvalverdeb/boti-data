from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from boti_data.datacube import DatacubeConfig, DatacubeResource
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.db.sql_resource import SqlDatabaseResource
from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource

BackendName = str
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
    duplicate_expr: Union[str, list[str]] | None = None
    duplicate_keep: Union[Literal["first", "last"], bool] = "last"
    group_by_expr: Union[str, list[str]] | None = None
    group_expr: Union[str, dict[str, str]] | None = None


class NormalizedFilters(BaseModel):
    """Result of normalizing configured-mode filters into control + filter buckets."""

    model_config = ConfigDict(extra="forbid")

    control: dict[str, Any] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)


class ConfiguredRequest(BaseModel):
    """Typed result of building a configured-mode SQL request.

    Carries the reflected SA model, prepared statement, DB-translated filters,
    control kwargs, and resolved fieldnames so callers don't need to recompute
    fieldnames from the control dict.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model: Any
    statement: Any
    db_filters: dict[str, Any] = Field(default_factory=dict)
    db_columns: list[str] | None = None
    control: dict[str, Any] = Field(default_factory=dict)
    configured_fieldnames: tuple[str, Any] | None = None


class PartitionedLoadConfig(BaseModel):
    """Typed configuration for building a partitioned SQL load request."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    statement: Any
    model: Any
    filters: dict[str, Any] = Field(default_factory=dict)
    partitioned: bool = True
    as_pandas: bool = False
    limit: int | None = None
    chunk_size: int | None = None
    single_fetch_threshold: int | None = None
    max_concurrent_fetches: int | None = None
    partition_strategy: str | None = None
    partition_column: str | None = None
    order_column: str | None = None
    diagnostics: bool = False


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

    @model_validator(mode="after")
    def validate_request(self) -> SqlLoadRequest:
        self._validate_source_selection()
        self._validate_projection_requirements()
        self._validate_filter_requirements()
        self._validate_return_mode()
        return self

    def _validate_source_selection(self) -> None:
        if not self.sql and self.statement is None:
            raise ValueError("Either sql or statement must be provided.")
        if self.sql and self.statement is not None:
            raise ValueError("Provide either sql or statement, not both.")

    def _validate_projection_requirements(self) -> None:
        if self.columns:
            if self.statement is None:
                raise ValueError("columns require a SQLAlchemy statement input.")
            if self.model is None:
                raise ValueError("columns require model for SQLAlchemy column resolution.")

    def _validate_filter_requirements(self) -> None:
        if self.filters:
            if self.statement is None:
                raise ValueError("filters require a SQLAlchemy statement input.")
            if self.model is None:
                raise ValueError("filters require model for SQLAlchemy column resolution.")

    def _validate_return_mode(self) -> None:
        if self.return_type == "dask":
            if self.statement is None or self.model is None:
                raise ValueError(
                    "return_type='dask' requires statement and model for lazy SQL path."
                )
        elif self.return_type == "pandas" and not self.as_pandas:
            raise ValueError(
                "SqlLoadRequest eager return types require as_pandas=True. Use DataGateway.load(...) "
                "with statement and model for the default lazy SQL path, or set as_pandas=True for eager convenience reads."
            )


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


@dataclass(frozen=True, slots=True)
class TableDescription:
    """Result of :meth:`~boti_data.gateway.core.DataGateway.describe`.

    ``row_count`` is exact when ``row_count_is_exact`` is ``True``; otherwise
    it reports the ``row_count_limit`` cap that was hit, meaning "at least
    this many rows" rather than the real total.
    """

    table: str
    columns: dict[str, str]
    row_count: int
    row_count_is_exact: bool
