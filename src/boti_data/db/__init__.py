"""
Database-backed data resources and helpers.
"""

from boti_data.db.partitioned_loader import SqlPartitionedLoader
from boti_data.db.partitioned_types import (
    SqlPartitionedLoadRequest,
    SqlPartitionPlan,
    SqlPartitionSpec,
)
from boti_data.db.sql_manager import (
    AsyncSqlDatabaseResource,
    EngineRegistry,
    SqlDatabaseConfig,
    SqlDatabaseResource,
)
from boti_data.db.sql_model_builder import BuilderConfig, SqlAlchemyModelBuilder
from boti_data.db.sql_model_registry import (
    DefaultBase,
    RegistryConfig,
    SqlModelRegistry,
    get_global_registry,
)
from boti_data.db.sqlalchemy_async import ensure_greenlet_available

__all__ = [
    "AsyncSqlDatabaseResource",
    "BuilderConfig",
    "DefaultBase",
    "EngineRegistry",
    "RegistryConfig",
    "SqlAlchemyModelBuilder",
    "SqlDatabaseConfig",
    "SqlDatabaseResource",
    "SqlPartitionPlan",
    "SqlPartitionSpec",
    "SqlPartitionedLoadRequest",
    "SqlPartitionedLoader",
    "SqlModelRegistry",
    "ensure_greenlet_available",
    "get_global_registry",
]
