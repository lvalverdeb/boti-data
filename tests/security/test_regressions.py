"""
Security regression tests for recently fixed audit findings.
"""

from __future__ import annotations

import pytest

from boti_data.db.sql_manager import AsyncSqlDatabaseResource, EngineRegistry, SqlDatabaseConfig, SqlDatabaseResource


pytestmark = pytest.mark.security_regression


def test_sync_engine_registry_separates_query_only_from_writable_configs():
    """Verify a writable engine is never reused for a query_only resource, or vice versa."""
    query_only_config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:pass@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
    )
    writable_config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:pass@localhost/test_db",
        query_only=False,
        poolclass="sqlalchemy.pool.NullPool",
    )

    readonly_resource = SqlDatabaseResource(query_only_config)
    writable_resource = SqlDatabaseResource(writable_config)

    try:
        assert readonly_resource._engine is not writable_resource._engine
        assert readonly_resource._engine_key != writable_resource._engine_key
    finally:
        readonly_resource.close()
        writable_resource.close()


@pytest.mark.asyncio
async def test_async_engine_registry_separates_query_only_from_writable_configs():
    """Verify async engine caching never mixes query_only and writable resources."""
    query_only_config = SqlDatabaseConfig(
        connection_url="mysql+asyncmy://user:pass@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
    )
    writable_config = SqlDatabaseConfig(
        connection_url="mysql+asyncmy://user:pass@localhost/test_db",
        query_only=False,
        poolclass="sqlalchemy.pool.NullPool",
    )

    async with AsyncSqlDatabaseResource(query_only_config) as readonly_resource:
        readonly_key = readonly_resource._engine_key
        readonly_engine = readonly_resource._engine

        async with AsyncSqlDatabaseResource(writable_config) as writable_resource:
            assert readonly_engine is not writable_resource._engine
            assert readonly_key != writable_resource._engine_key


def test_query_only_and_writable_registry_entries_can_coexist_without_collision():
    """Verify registry bookkeeping keeps readonly and writable entries isolated."""
    query_only_config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:pass@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
    )
    writable_config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:pass@localhost/test_db",
        query_only=False,
        poolclass="sqlalchemy.pool.NullPool",
    )

    readonly_resource = SqlDatabaseResource(query_only_config)
    writable_resource = SqlDatabaseResource(writable_config)

    try:
        assert readonly_resource._engine_key in EngineRegistry._registry
        assert writable_resource._engine_key in EngineRegistry._registry
        assert readonly_resource._engine_key != writable_resource._engine_key
    finally:
        readonly_resource.close()
        writable_resource.close()
