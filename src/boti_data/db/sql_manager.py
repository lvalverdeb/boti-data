"""
Database configuration, engine helpers, and managed SQL resources.
"""

from boti_data.db.engine_registry import EngineRegistry
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.db.sql_engine import (
    _build_async_engine_key,
    _build_async_engine_kwargs,
    _build_driver_example,
    _build_sync_engine_key,
    _build_sync_engine_kwargs,
    _configure_query_only_engine,
    _create_worker_sync_engine,
    _get_worker_engine_identity,
    _render_normalized_url,
    _validate_async_driver_url,
    _validate_query_only_support,
    _validate_sync_driver_url,
)
from boti_data.db.sql_readonly import ReadOnlyAsyncSession, ReadOnlySession
from boti_data.db.sql_resource import AsyncSqlDatabaseResource, SqlDatabaseResource

__all__ = [
    "AsyncSqlDatabaseResource",
    "EngineRegistry",
    "ReadOnlyAsyncSession",
    "ReadOnlySession",
    "SqlDatabaseConfig",
    "SqlDatabaseResource",
    "_build_async_engine_key",
    "_build_async_engine_kwargs",
    "_build_driver_example",
    "_build_sync_engine_key",
    "_build_sync_engine_kwargs",
    "_configure_query_only_engine",
    "_create_worker_sync_engine",
    "_get_worker_engine_identity",
    "_render_normalized_url",
    "_validate_async_driver_url",
    "_validate_query_only_support",
    "_validate_sync_driver_url",
]
