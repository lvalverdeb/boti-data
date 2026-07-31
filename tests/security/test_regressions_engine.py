"""
Security regression tests: engine registry isolation and worker credential
serialisation.

Split out of test_regressions.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import pytest
from boti.core.logger import Logger

from boti_data.db.sql_config import SqlDatabaseConfig, WorkerSqlConfig
from boti_data.db.sql_manager import AsyncSqlDatabaseResource, EngineRegistry, SqlDatabaseResource

pytestmark = pytest.mark.security_regression


def test_sync_engine_registry_separates_query_only_from_writable_configs() -> None:
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
async def test_async_engine_registry_separates_query_only_from_writable_configs() -> None:
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


def test_query_only_and_writable_registry_entries_can_coexist_without_collision() -> None:
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


def test_worker_config_uses_env_var_when_set() -> None:
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


def test_worker_config_warns_when_no_env_var_set(monkeypatch) -> None:
    """WorkerSqlConfig.from_database_config must log a WARNING via boti's
    structured Logger when falling back to raw DSN -- not warnings.warn(),
    which is too easy to filter/suppress for a security-relevant condition
    (real credentials landing in every worker task payload)."""
    config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:secret@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
    )
    logger = Logger.default_logger(logger_name="WorkerSqlConfig")
    warnings_seen: list[str] = []
    monkeypatch.setattr(logger, "warning", lambda msg, *a, **kw: warnings_seen.append(msg))

    worker = WorkerSqlConfig.from_database_config(config)

    assert any("worker_connection_env_var" in msg for msg in warnings_seen), (
        "Expected a structured WARNING log about worker_connection_env_var not being set"
    )
    assert worker.connection_url is not None  # fallback still works


def test_worker_config_raw_dsn_fallback_carries_credentials() -> None:
    """Sanity-check: when no env var is configured the raw DSN is present (so callers are aware)."""
    config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:secret@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
    )
    worker = WorkerSqlConfig.from_database_config(config)

    assert worker.connection_env_var is None
    secret_val = worker.connection_url.get_secret_value()  # type: ignore[union-attr]
    assert "secret" in secret_val


def test_engine_keys_differ_when_only_the_password_differs() -> None:
    """Two configs identical except for the password must never alias to the
    same cached engine (e.g. after a credential rotation), and the key itself
    must not contain the plaintext secret."""
    from pydantic import SecretStr

    from boti_data.db.sql_engine import _build_async_engine_key, _build_sync_engine_key

    config_a = SqlDatabaseConfig(
        connection_url=SecretStr("mysql+pymysql://app:old-password@db.internal:3306/reports")
    )
    config_b = SqlDatabaseConfig(
        connection_url=SecretStr("mysql+pymysql://app:new-password@db.internal:3306/reports")
    )

    sync_key_a = _build_sync_engine_key(config_a)
    sync_key_b = _build_sync_engine_key(config_b)
    assert sync_key_a != sync_key_b

    async_key_a = _build_async_engine_key(config_a)
    async_key_b = _build_async_engine_key(config_b)
    assert async_key_a != async_key_b

    for key in (sync_key_a, sync_key_b, async_key_a, async_key_b):
        assert "old-password" not in repr(key)
        assert "new-password" not in repr(key)

    # Same config still produces a stable key (cache hits keep working).
    assert sync_key_a == _build_sync_engine_key(config_a)
