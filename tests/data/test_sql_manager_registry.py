"""
Tests for SQL configuration loading, engine registry caching/locking, and
DSN/driver validation.

Split out of test_sql_manager.py purely for god-module/long-file headroom.
"""

import pytest
from pydantic import ValidationError
from sqlalchemy.engine import url as sqlalchemy_url
from sqlalchemy.exc import SQLAlchemyError

from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.db.sql_manager import (
    AsyncSqlDatabaseResource,
    EngineRegistry,
    SqlDatabaseResource,
    _build_sync_engine_kwargs,
    _validate_query_only_support,
)
from boti_data.db.sqlalchemy_async import ensure_greenlet_available


def test_config_validation() -> None:
    """Verify DSN and initialization constraints are respected."""
    # Invalid: missing URL
    with pytest.raises(ValidationError):
        SqlDatabaseConfig()

    # Valid config
    config = SqlDatabaseConfig(connection_url="sqlite:///:memory:")
    assert config.pool_pre_ping is True
    assert config.query_only is True


def test_config_from_env_uses_pydantic_settings(monkeypatch) -> None:
    """Verify env-backed DB settings load through pydantic-settings."""
    monkeypatch.setenv("DB_CONNECTION_URL", "sqlite:///:memory:")
    monkeypatch.setenv("DB_POOL_SIZE", "9")
    monkeypatch.setenv("DB_QUERY_ONLY", "false")

    config = SqlDatabaseConfig.from_env()

    assert config.connection_url.get_secret_value() == "sqlite:///:memory:"
    assert config.pool_size == 9
    assert config.query_only is False


def test_config_from_env_file_uses_pydantic_settings(tmp_path) -> None:
    """Verify dotenv-backed DB settings load through pydantic-settings."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DB_CONNECTION_URL='sqlite:///:memory:'\nDB_POOL_TIMEOUT=41\n",
        encoding="utf-8",
    )

    config = SqlDatabaseConfig.from_env(env_file=env_file)

    assert config.connection_url.get_secret_value() == "sqlite:///:memory:"
    assert config.pool_timeout == 41


def test_config_from_named_env_prefix(tmp_path) -> None:
    """Verify multiple DB profiles can be loaded from explicit prefixes."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "PRIMARY_DB_CONNECTION_URL='sqlite:///:memory:'",
                "PRIMARY_DB_QUERY_ONLY=false",
                "PRIMARY_DB_POOL_SIZE=7",
                "ANALYTICS_DB_CONNECTION_URL='sqlite:///:memory:'",
                "ANALYTICS_DB_QUERY_ONLY=true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    primary = SqlDatabaseConfig.from_env_prefix("PRIMARY_DB_", env_file=env_file)
    analytics = SqlDatabaseConfig.from_env_prefix("ANALYTICS_DB_", env_file=env_file)

    assert primary.pool_size == 7
    assert primary.query_only is False
    assert analytics.query_only is True


def test_config_rejects_non_allowlisted_poolclass_import() -> None:
    """Verify poolclass import strings are restricted to a safe allowlist."""
    with pytest.raises(ValidationError, match="poolclass must be one of"):
        SqlDatabaseConfig(
            connection_url="sqlite:///:memory:",
            poolclass="subprocess.Popen",
        )


def test_engine_registry_reference_counting() -> None:
    """Verify the registry correctly shares and discards connection pools."""
    config = SqlDatabaseConfig(connection_url="sqlite:///:memory:", query_only=False)

    # Track the underlying key using the config
    # To properly simulate duplicate configs, we create two resources
    res1 = SqlDatabaseResource(config)
    res2 = SqlDatabaseResource(config)

    assert res1._engine is res2._engine

    key = res1._engine_key
    assert EngineRegistry._registry[key]["ref_count"] == 2

    # Close first resource
    res1.close()
    assert key in EngineRegistry._registry
    assert EngineRegistry._registry[key]["ref_count"] == 1

    # Close second resource
    res2.close()
    assert key not in EngineRegistry._registry


def test_engine_registry_creates_sync_engine_outside_registry_lock(monkeypatch) -> None:
    if not hasattr(EngineRegistry._lock, "_is_owned"):
        pytest.skip("RLock ownership inspection is unavailable on this runtime")

    class DummyEngine:
        def dispose(self) -> None:
            pass

    owned_states = []

    def fake_create_engine(_url: str, **_kwargs) -> DummyEngine:
        owned_states.append(EngineRegistry._lock._is_owned())
        return DummyEngine()

    key = ("sync-lock-test",)
    monkeypatch.setattr("boti_data.db.engine_registry.create_engine", fake_create_engine)

    try:
        _engine, reused = EngineRegistry.get_or_create(key, "sqlite://")
        assert reused is False
        assert owned_states == [False]
    finally:
        EngineRegistry.release(key)


@pytest.mark.asyncio
async def test_engine_registry_creates_async_engine_outside_registry_lock(monkeypatch) -> None:
    if not hasattr(EngineRegistry._lock, "_is_owned"):
        pytest.skip("RLock ownership inspection is unavailable on this runtime")

    class DummyAsyncEngine:
        async def dispose(self) -> None:
            return None

    owned_states = []

    def fake_create_async_engine(_url: str, **_kwargs) -> DummyAsyncEngine:
        owned_states.append(EngineRegistry._lock._is_owned())
        return DummyAsyncEngine()

    key = ("async-lock-test",)
    monkeypatch.setattr(
        "boti_data.db.engine_registry.create_async_engine", fake_create_async_engine
    )

    try:
        _engine, reused = await EngineRegistry.get_or_create_async(key, "sqlite+aiosqlite://")
        assert reused is False
        assert owned_states == [False]
    finally:
        await EngineRegistry.release_async(key)


def test_sync_resource_rejects_async_dsn_with_actionable_error() -> None:
    """Verify synchronous resources fail fast on async-only drivers."""
    config = SqlDatabaseConfig(connection_url="mysql+asyncmy://user:pass@localhost/test_db")

    with pytest.raises(SQLAlchemyError, match="only supports synchronous SQLAlchemy drivers"):
        SqlDatabaseResource(config)


def test_sync_resource_rejects_missing_driver_with_actionable_error() -> None:
    """A syntactically valid DSN whose DBAPI driver package isn't installed must
    fail fast with a clear message naming the missing package, instead of only
    surfacing deep inside create_engine()/first connect(). boti-data does not
    declare a Postgres driver dependency, so psycopg2 is reliably absent here."""
    config = SqlDatabaseConfig(connection_url="postgresql+psycopg2://user:pass@localhost/test_db")

    with pytest.raises(SQLAlchemyError, match="driver package is not installed"):
        SqlDatabaseResource(config)


def test_sync_resource_normalizes_common_mysql_driver_alias() -> None:
    """Verify legacy mysql+pymsql DSNs are coerced to the supported pymysql driver."""
    config = SqlDatabaseConfig(
        connection_url="mysql+pymsql://user:pass@localhost/test_db",
        query_only=False,
        poolclass="sqlalchemy.pool.NullPool",
    )

    db = SqlDatabaseResource(config)

    try:
        assert db.engine.url.drivername == "mysql+pymysql"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_async_resource_rejects_sync_dsn_with_actionable_error() -> None:
    """Verify async resources fail fast on synchronous drivers."""
    config = SqlDatabaseConfig(connection_url="mysql+pymysql://user:pass@localhost/test_db")

    with pytest.raises(SQLAlchemyError, match="requires an asynchronous SQLAlchemy driver"):
        async with AsyncSqlDatabaseResource(config):
            pass


def test_async_sqlalchemy_requires_greenlet(monkeypatch) -> None:
    """Verify missing greenlet raises a clear installation error."""
    monkeypatch.setattr("boti_data.db.sqlalchemy_async.find_spec", lambda name: None)

    with pytest.raises(SQLAlchemyError, match="require the 'greenlet' package"):
        ensure_greenlet_available()


def test_sync_resource_accepts_clickhouse_dsn() -> None:
    """Verify SqlDatabaseResource resolves the clickhouse-connect dialect."""
    config = SqlDatabaseConfig(
        connection_url="clickhousedb://user:pass@localhost:8123/test_db",
        query_only=False,
        poolclass="sqlalchemy.pool.NullPool",
    )

    db = SqlDatabaseResource(config)

    try:
        assert db.engine.url.get_backend_name() == "clickhousedb"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_async_resource_rejects_clickhouse_dsn_with_actionable_error() -> None:
    """ClickHouse has no maintained async SQLAlchemy driver, so the async resource
    must reject it up front instead of failing deep inside engine construction."""
    config = SqlDatabaseConfig(connection_url="clickhousedb://user:pass@localhost:8123/test_db")

    with pytest.raises(SQLAlchemyError, match="does not support ClickHouse"):
        async with AsyncSqlDatabaseResource(config):
            pass


def test_validate_query_only_support_accepts_clickhouse() -> None:
    """Verify query_only=True is allowed for clickhouse, enforced via the driver's
    native readonly setting rather than a Postgres/MySQL-style SET statement."""
    parsed = sqlalchemy_url.make_url("clickhousedb://user:pass@localhost:8123/test_db")

    _validate_query_only_support(parsed)


def test_sync_resource_accepts_duckdb_dsn(tmp_path) -> None:
    """Verify SqlDatabaseResource resolves the duckdb-engine dialect."""
    config = SqlDatabaseConfig(
        connection_url=f"duckdb:///{tmp_path / 'test.duckdb'}",
        query_only=False,
        poolclass="sqlalchemy.pool.NullPool",
    )

    db = SqlDatabaseResource(config)

    try:
        assert db.engine.url.get_backend_name() == "duckdb"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_async_resource_rejects_duckdb_dsn_with_actionable_error(tmp_path) -> None:
    """DuckDB has no maintained async SQLAlchemy driver, so the async resource
    must reject it up front instead of failing deep inside engine construction."""
    config = SqlDatabaseConfig(connection_url=f"duckdb:///{tmp_path / 'test.duckdb'}")

    with pytest.raises(SQLAlchemyError, match="does not support DuckDB"):
        async with AsyncSqlDatabaseResource(config):
            pass


def test_validate_query_only_support_accepts_duckdb(tmp_path) -> None:
    """Verify query_only=True is allowed for duckdb — enforced via a connect-time
    read_only flag rather than a Postgres/MySQL-style post-connect SET statement."""
    parsed = sqlalchemy_url.make_url(f"duckdb:///{tmp_path / 'test.duckdb'}")

    _validate_query_only_support(parsed)


def test_build_sync_engine_kwargs_injects_duckdb_read_only(tmp_path) -> None:
    """Verify query_only=True causes connect_args={"read_only": True} to be
    injected for duckdb — this is the actual enforcement mechanism, since
    DuckDB's read-only mode is a connect-time flag rather than something
    settable after the engine already exists."""
    config = SqlDatabaseConfig(
        connection_url=f"duckdb:///{tmp_path / 'test.duckdb'}",
        query_only=True,
    )

    kwargs = _build_sync_engine_kwargs(config)

    assert kwargs["connect_args"] == {"read_only": True}


def test_build_sync_engine_kwargs_does_not_inject_read_only_when_query_only_false(
    tmp_path,
) -> None:
    """Verify query_only=False leaves connect_args untouched for duckdb."""
    config = SqlDatabaseConfig(
        connection_url=f"duckdb:///{tmp_path / 'test.duckdb'}",
        query_only=False,
    )

    kwargs = _build_sync_engine_kwargs(config)

    assert kwargs["connect_args"] == {}


def test_build_sync_engine_kwargs_does_not_inject_read_only_for_other_backends(tmp_path) -> None:
    """Verify the duckdb-specific connect_args injection never fires for a
    backend that doesn't need it, even with query_only=True."""
    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{tmp_path / 'test.db'}",
        query_only=True,
    )

    kwargs = _build_sync_engine_kwargs(config)

    assert kwargs["connect_args"] == {}
