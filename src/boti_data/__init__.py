"""
Data modules and interfaces for the Boti pipeline context.
"""

from boti_data.connection_catalog import ConnectionCatalog
from boti_data.db import (
    AsyncSqlDatabaseResource,
    BuilderConfig,
    DefaultBase,
    EngineRegistry,
    RegistryConfig,
    SqlAlchemyModelBuilder,
    SqlDatabaseConfig,
    SqlDatabaseResource,
    SqlModelRegistry,
    SqlPartitionedLoader,
    SqlPartitionedLoadRequest,
    SqlPartitionPlan,
    SqlPartitionSpec,
    ensure_greenlet_available,
    get_global_registry,
)
from boti_data.field_map import FieldMap
from boti_data.filters import (
    And,
    Expr,
    FilterHandler,
    Not,
    Or,
    TrueExpr,
)
# parquet must be imported before gateway to avoid a circular import:
# gateway.core → parquet.resource → parquet/__init__ → parquet.reader → gateway.core
from boti_data.parquet import ParquetDataConfig, ParquetDataResource, ParquetReader
from boti_data.gateway import (
    DataFrameOptions,
    DataFrameParams,
    DataGateway,
    ParquetLoadRequest,
    SqlLoadRequest,
)
from boti_data.helper import DataHelper
from boti_data.joins import indexed_left_join, left_join_frames
from boti_data.schema import (
    SchemaValidationError,
    align_frames_for_join,
    apply_schema_map,
    infer_schema_map,
    normalize_dtype_alias,
    normalize_schema_map,
    validate_schema,
)


__all__ = [
    "And",
    "AsyncSqlDatabaseResource",
    "BuilderConfig",
    "ConnectionCatalog",
    "DataFrameOptions",
    "DataFrameParams",
    "DataGateway",
    "DataHelper",
    "DefaultBase",
    "EngineRegistry",
    "Expr",
    "FieldMap",
    "FilterHandler",
    "indexed_left_join",
    "Not",
    "Or",
    "ParquetDataConfig",
    "ParquetLoadRequest",
    "ParquetDataResource",
    "ParquetReader",
    "RegistryConfig",
    "SchemaValidationError",
    "SqlLoadRequest",
    "SqlAlchemyModelBuilder",
    "SqlDatabaseConfig",
    "SqlDatabaseResource",
    "SqlPartitionPlan",
    "SqlPartitionSpec",
    "SqlPartitionedLoadRequest",
    "SqlPartitionedLoader",
    "SqlModelRegistry",
    "TrueExpr",
    "align_frames_for_join",
    "apply_schema_map",
    "ensure_greenlet_available",
    "get_global_registry",
    "infer_schema_map",
    "left_join_frames",
    "normalize_dtype_alias",
    "normalize_schema_map",
    "validate_schema",
]
