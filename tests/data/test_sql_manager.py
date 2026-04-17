"""
Tests for the SQLAlchemy connection manager and engine registry.
"""
import pickle

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm.exc import UnmappedInstanceError

from boti_data.db.sql_config import WorkerSqlConfig
from boti_data.db.sql_manager import (
    AsyncSqlDatabaseResource,
    EngineRegistry,
    SqlDatabaseConfig,
    SqlDatabaseResource,
    _create_worker_sync_engine,
)
from boti_data.db.sqlalchemy_async import ensure_greenlet_available


def test_config_validation():
    """Verify DSN and initialization constraints are respected."""
    # Invalid: missing URL
    with pytest.raises(ValidationError):
        SqlDatabaseConfig()

    # Valid config
    config = SqlDatabaseConfig(connection_url="sqlite:///:memory:")
    assert config.pool_pre_ping is True
    assert config.query_only is True


def test_config_from_env_uses_pydantic_settings(monkeypatch):
    """Verify env-backed DB settings load through pydantic-settings."""
    monkeypatch.setenv("DB_CONNECTION_URL", "sqlite:///:memory:")
    monkeypatch.setenv("DB_POOL_SIZE", "9")
    monkeypatch.setenv("DB_QUERY_ONLY", "false")

    config = SqlDatabaseConfig.from_env()

    assert config.connection_url.get_secret_value() == "sqlite:///:memory:"
    assert config.pool_size == 9
    assert config.query_only is False


def test_config_from_env_file_uses_pydantic_settings(tmp_path):
    """Verify dotenv-backed DB settings load through pydantic-settings."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DB_CONNECTION_URL='sqlite:///:memory:'\nDB_POOL_TIMEOUT=41\n",
        encoding="utf-8",
    )

    config = SqlDatabaseConfig.from_env(env_file=env_file)

    assert config.connection_url.get_secret_value() == "sqlite:///:memory:"
    assert config.pool_timeout == 41


def test_config_from_named_env_prefix(tmp_path):
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


def test_config_rejects_non_allowlisted_poolclass_import():
    """Verify poolclass import strings are restricted to a safe allowlist."""
    with pytest.raises(ValidationError, match="poolclass must be one of"):
        SqlDatabaseConfig(
            connection_url="sqlite:///:memory:",
            poolclass="subprocess.Popen",
        )


def test_resource_initializes_engine_correctly():
    """Verify the context manager builds the engine securely."""
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with SqlDatabaseResource(config) as db:
        assert db.engine is not None
        assert db.session is not None

        # Execute a simple DB Ping via connection
        with db.engine.connect() as conn:
            assert conn is not None

        with db.session() as session:
            assert session.execute(text("SELECT 1")).scalar() == 1


def test_engine_registry_reference_counting():
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


def test_engine_registry_creates_sync_engine_outside_registry_lock(monkeypatch):
    if not hasattr(EngineRegistry._lock, "_is_owned"):
        pytest.skip("RLock ownership inspection is unavailable on this runtime")

    class DummyEngine:
        def dispose(self) -> None:
            pass

    owned_states = []

    def fake_create_engine(_url: str, **_kwargs):
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
async def test_engine_registry_creates_async_engine_outside_registry_lock(monkeypatch):
    if not hasattr(EngineRegistry._lock, "_is_owned"):
        pytest.skip("RLock ownership inspection is unavailable on this runtime")

    class DummyAsyncEngine:
        async def dispose(self) -> None:
            return None

    owned_states = []

    def fake_create_async_engine(_url: str, **_kwargs):
        owned_states.append(EngineRegistry._lock._is_owned())
        return DummyAsyncEngine()

    key = ("async-lock-test",)
    monkeypatch.setattr("boti_data.db.engine_registry.create_async_engine", fake_create_async_engine)

    try:
        _engine, reused = await EngineRegistry.get_or_create_async(key, "sqlite+aiosqlite://")
        assert reused is False
        assert owned_states == [False]
    finally:
        await EngineRegistry.release_async(key)


def test_resource_cleanup_failure_resilience():
    """Verify resource cleanup is gracefully handled and engine is destroyed."""
    config = SqlDatabaseConfig(connection_url="sqlite:///:memory:", query_only=False)
    db = SqlDatabaseResource(config)
    engine = db.engine
    db.close()

    with pytest.raises(RuntimeError, match="is closed"):
        _ = db.engine

    assert engine.pool is not None  # It was active before, now detached safely


def test_sync_resource_rejects_async_dsn_with_actionable_error():
    """Verify synchronous resources fail fast on async-only drivers."""
    config = SqlDatabaseConfig(connection_url="mysql+asyncmy://user:pass@localhost/test_db")

    with pytest.raises(SQLAlchemyError, match="only supports synchronous SQLAlchemy drivers"):
        SqlDatabaseResource(config)


def test_sync_resource_normalizes_common_mysql_driver_alias():
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
async def test_async_resource_rejects_sync_dsn_with_actionable_error():
    """Verify async resources fail fast on synchronous drivers."""
    config = SqlDatabaseConfig(connection_url="mysql+pymysql://user:pass@localhost/test_db")

    with pytest.raises(SQLAlchemyError, match="requires an asynchronous SQLAlchemy driver"):
        async with AsyncSqlDatabaseResource(config):
            pass


def test_async_sqlalchemy_requires_greenlet(monkeypatch):
    """Verify missing greenlet raises a clear installation error."""
    monkeypatch.setattr("boti_data.db.sqlalchemy_async.find_spec", lambda name: None)

    with pytest.raises(SQLAlchemyError, match="require the 'greenlet' package"):
        ensure_greenlet_available()


def test_sync_resource_defaults_to_query_only():
    """Verify synchronous query_only relies on native database-enforced read-only access."""
    # Prepare a writable SQLite database first, then reopen it through a read-only DSN.
    # SQLAlchemy requires URI mode for SQLite read-only connections.
    from sqlalchemy import create_engine

    def create_db(path):
        engine = create_engine(f"sqlite:///{path}")
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE readonly_table (id INTEGER PRIMARY KEY, name TEXT)")
            conn.exec_driver_sql("INSERT INTO readonly_table (name) VALUES ('Alice')")
        engine.dispose()

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "readonly.db"
        create_db(db_path)
        read_only_url = f"sqlite:///file:{db_path}?mode=ro&uri=true"
        config = SqlDatabaseConfig(connection_url=read_only_url, poolclass="sqlalchemy.pool.NullPool")

        with SqlDatabaseResource(config) as db, db.session() as session:
            with db.engine.connect() as conn:
                assert conn.execute(text("SELECT name FROM readonly_table")).scalar() == "Alice"
                with pytest.raises(OperationalError, match="readonly"):
                    conn.exec_driver_sql("INSERT INTO readonly_table (name) VALUES ('Bob')")

            with pytest.raises(SQLAlchemyError, match="read-only"):
                session.add(object())

            with pytest.raises(OperationalError, match="readonly"):
                session.execute(text("DELETE FROM readonly_table"))

            with pytest.raises(SQLAlchemyError, match="read-only"):
                session.commit()


def test_query_only_rejects_sqlite_memory_without_native_read_only():
    """Verify query_only=True rejects SQLite DSNs that cannot enforce read-only mode."""
    config = SqlDatabaseConfig(connection_url="sqlite:///:memory:", poolclass="sqlalchemy.pool.StaticPool")

    with pytest.raises(SQLAlchemyError, match="cannot enforce read-only mode"):
        SqlDatabaseResource(config)


@pytest.mark.asyncio
async def test_async_resource_defaults_to_query_only():
    """Verify async sessions remain read-only by default."""
    config = SqlDatabaseConfig(connection_url="mysql+asyncmy://user:pass@localhost/test_db")

    async with AsyncSqlDatabaseResource(config) as db:
        session = db.session()

        with pytest.raises(SQLAlchemyError, match="read-only"):
            session.add(object())

        with pytest.raises(SQLAlchemyError, match="read-only"):
            await session.commit()


def test_query_only_false_allows_sync_mutations():
    """Verify explicitly disabling query-only mode restores write capabilities."""
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with SqlDatabaseResource(config) as db:
        with db.engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE writable_table (id INTEGER PRIMARY KEY, name TEXT)")
            conn.exec_driver_sql("INSERT INTO writable_table (id, name) VALUES (1, 'Alice')")

        with db.session() as session:
            rows = session.execute(text("SELECT name FROM writable_table")).scalars().all()

        assert rows == ["Alice"]


def test_worker_sync_engine_preserves_query_only(tmp_path):
    """Verify worker-local engines keep native database read-only enforcement."""
    from sqlalchemy import create_engine

    db_path = tmp_path / "worker_readonly.db"
    bootstrap_engine = create_engine(f"sqlite:///{db_path}")
    try:
        with bootstrap_engine.begin() as conn:
            conn.exec_driver_sql(
                "CREATE TABLE readonly_table (id INTEGER PRIMARY KEY, name TEXT)"
            )
            conn.exec_driver_sql(
                "INSERT INTO readonly_table (id, name) VALUES (1, 'Alice')"
            )
    finally:
        bootstrap_engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///file:{db_path}?mode=ro&uri=true",
        poolclass="sqlalchemy.pool.NullPool",
    )

    engine = _create_worker_sync_engine(config)
    try:
        with engine.connect() as conn:
            assert conn.execute(text("SELECT name FROM readonly_table")).scalar() == "Alice"
            with pytest.raises(OperationalError, match="readonly"):
                conn.exec_driver_sql(
                    "INSERT INTO readonly_table (id, name) VALUES (2, 'Bob')"
                )
    finally:
        engine.dispose()


def test_sql_database_resource_is_pickleable(tmp_path):
    """Verify SqlDatabaseResource can round-trip through pickle and restore its engine."""
    db_path = tmp_path / "pickleable.db"
    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        allow_pickle=True,
    )

    db = SqlDatabaseResource(config)
    restored = None
    try:
        with db.engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE pickle_table (id INTEGER PRIMARY KEY, name TEXT)")
            conn.exec_driver_sql("INSERT INTO pickle_table (id, name) VALUES (1, 'Alice')")

        with SqlDatabaseResource.trusted_unpickle_scope():
            restored = pickle.loads(pickle.dumps(db))

        with restored.session() as session:
            rows = session.execute(text("SELECT name FROM pickle_table")).scalars().all()

        assert rows == ["Alice"]
    finally:
        db.close()
        if restored is not None:
            restored.close()


def test_worker_sql_config_can_resolve_connection_from_env(monkeypatch, tmp_path):
    """Verify worker SQL payloads can avoid serializing DSNs by resolving them from env."""
    db_path = tmp_path / "worker-env.db"
    bootstrap = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        worker_connection_env_var="BOTI_WORKER_DSN",
    )

    payload = WorkerSqlConfig.from_database_config(bootstrap)
    assert payload.connection_url is None
    assert payload.connection_env_var == "BOTI_WORKER_DSN"

    monkeypatch.setenv("BOTI_WORKER_DSN", f"sqlite:///{db_path}")

    engine = _create_worker_sync_engine(
        payload.model_copy(
            update={"connection_env_var": "BOTI_WORKER_DSN"}
        )
    )
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE env_worker_table (id INTEGER PRIMARY KEY, name TEXT)")
            conn.exec_driver_sql("INSERT INTO env_worker_table (id, name) VALUES (1, 'Alice')")
        with engine.connect() as conn:
            assert conn.execute(text("SELECT name FROM env_worker_table")).scalar() == "Alice"
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_query_only_false_disables_async_session_guard():
    """Verify async query-only mode can be explicitly disabled."""
    config = SqlDatabaseConfig(
        connection_url="mysql+asyncmy://user:pass@localhost/test_db",
        query_only=False,
    )

    async with AsyncSqlDatabaseResource(config) as db:
        session = db.session()

        with pytest.raises(UnmappedInstanceError):
            session.add(object())
