"""
Security regression tests for recently fixed audit findings.
"""

from __future__ import annotations

import warnings

import pytest

from boti_data.db.sql_config import SqlDatabaseConfig, WorkerSqlConfig
from boti_data.db.sql_manager import AsyncSqlDatabaseResource, EngineRegistry, SqlDatabaseResource

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


# ---------------------------------------------------------------------------
# Worker credential serialisation tests
# ---------------------------------------------------------------------------


def test_worker_config_uses_env_var_when_set():
    """WorkerSqlConfig must NOT carry the raw DSN when worker_connection_env_var is configured."""
    config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:secret@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
        worker_connection_env_var="DB_DSN",
    )
    worker = WorkerSqlConfig.from_database_config(config)
    assert worker.connection_env_var == "DB_DSN"
    assert worker.connection_url is None, "DSN must not be serialized when env var is set"


def test_worker_config_warns_when_no_env_var_set():
    """WorkerSqlConfig.from_database_config must emit a UserWarning when falling back to raw DSN."""
    config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:secret@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        worker = WorkerSqlConfig.from_database_config(config)

    credential_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "worker_connection_env_var" in str(w.message)
    ]
    assert credential_warnings, "Expected a UserWarning about worker_connection_env_var not being set"
    assert worker.connection_url is not None  # fallback still works


def test_worker_config_raw_dsn_fallback_carries_credentials():
    """Sanity-check: when no env var is configured the raw DSN is present (so callers are aware)."""
    config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:secret@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        worker = WorkerSqlConfig.from_database_config(config)

    assert worker.connection_env_var is None
    secret_val = worker.connection_url.get_secret_value()  # type: ignore[union-attr]
    assert "secret" in secret_val

