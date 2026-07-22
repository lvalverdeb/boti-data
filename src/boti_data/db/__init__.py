"""
Database-backed data resources and helpers.
"""

from __future__ import annotations

import importlib
from typing import Any

from boti_data.db.partitioned_types import SqlPartitionPlan, SqlPartitionSpec
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
from boti_data.db.sql_repository import (
    AsyncSqlRepository,
    AsyncSqlUnitOfWork,
    SqlRepository,
    SqlUnitOfWork,
)
from boti_data.db.sqlalchemy_async import ensure_greenlet_available
from boti_data.db.vector_search import nearest_neighbors, vector_distance

__all__ = [
    "AsyncSqlDatabaseResource",
    "AsyncSqlRepository",
    "AsyncSqlUnitOfWork",
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
    "SqlRepository",
    "SqlUnitOfWork",
    "ensure_greenlet_available",
    "get_global_registry",
    "nearest_neighbors",
    "vector_distance",
]

# SqlPartitionedLoader/SqlPartitionedLoadRequest are the only heavy (dask+pandas)
# imports in this package — deferred so `from boti_data.db import SqlRepository`
# et al. doesn't pull them in.
_LAZY = {
    "SqlPartitionedLoader": ("boti_data.db.partitioned_loader", "SqlPartitionedLoader"),
    "SqlPartitionedLoadRequest": ("boti_data.db.partitioned_loader", "SqlPartitionedLoadRequest"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
