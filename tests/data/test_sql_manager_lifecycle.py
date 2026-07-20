"""
Tests for SqlDatabaseResource/AsyncSqlDatabaseResource lifecycle, query-only
enforcement, pickling, and worker-config resolution.

Split out of test_sql_manager.py purely for god-module/long-file headroom.
"""

import asyncio
import pickle

import pytest
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


class _SlowDisposeEngine:
    """Fake async engine whose ``dispose()`` suspends via a real await point,
    exposing TOCTOU races that a no-await fake cannot (asyncio only switches
    tasks at genuine suspension points)."""

    sync_engine = object()
    dispose_calls = 0

    async def dispose(self) -> None:
        await asyncio.sleep(0.05)
        _SlowDisposeEngine.dispose_calls += 1


def _make_fake_get_or_create_async(engine_key: tuple, shared_engine: _SlowDisposeEngine):
    async def fake_get_or_create_async(*_args, **_kwargs) -> tuple[_SlowDisposeEngine, bool]:
        # Register this resource's own reference (ref_count=1), the way the
        # real EngineRegistry.get_or_create_async would.
        with EngineRegistry._lock:
            EngineRegistry._registry[engine_key] = {"engine": shared_engine, "ref_count": 1}
        return shared_engine, False

    return fake_get_or_create_async


def test_resource_initializes_engine_correctly() -> None:
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


def test_resource_cleanup_failure_resilience() -> None:
    """Verify resource cleanup is gracefully handled and engine is destroyed."""
    config = SqlDatabaseConfig(connection_url="sqlite:///:memory:", query_only=False)
    db = SqlDatabaseResource(config)
    engine = db.engine
    db.close()

    with pytest.raises(RuntimeError, match="is closed"):
        _ = db.engine

    assert engine.pool is not None  # It was active before, now detached safely


def test_sync_resource_defaults_to_query_only() -> None:
    """Verify synchronous query_only relies on native database-enforced read-only access."""
    # Prepare a writable SQLite database first, then reopen it through a read-only DSN.
    # SQLAlchemy requires URI mode for SQLite read-only connections.
    from sqlalchemy import create_engine

    def create_db(path) -> None:
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
        config = SqlDatabaseConfig(
            connection_url=read_only_url, poolclass="sqlalchemy.pool.NullPool"
        )

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


def test_query_only_rejects_sqlite_memory_without_native_read_only() -> None:
    """Verify query_only=True rejects SQLite DSNs that cannot enforce read-only mode."""
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:", poolclass="sqlalchemy.pool.StaticPool"
    )

    with pytest.raises(SQLAlchemyError, match="cannot enforce read-only mode"):
        SqlDatabaseResource(config)


@pytest.mark.asyncio
async def test_async_resource_defaults_to_query_only() -> None:
    """Verify async sessions remain read-only by default."""
    config = SqlDatabaseConfig(connection_url="mysql+asyncmy://user:pass@localhost/test_db")

    async with AsyncSqlDatabaseResource(config) as db:
        session = db.session()

        with pytest.raises(SQLAlchemyError, match="read-only"):
            session.add(object())

        with pytest.raises(SQLAlchemyError, match="read-only"):
            await session.commit()


def test_query_only_false_allows_sync_mutations() -> None:
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


def test_worker_sync_engine_preserves_query_only(tmp_path) -> None:
    """Verify worker-local engines keep native database read-only enforcement."""
    from sqlalchemy import create_engine

    db_path = tmp_path / "worker_readonly.db"
    bootstrap_engine = create_engine(f"sqlite:///{db_path}")
    try:
        with bootstrap_engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE readonly_table (id INTEGER PRIMARY KEY, name TEXT)")
            conn.exec_driver_sql("INSERT INTO readonly_table (id, name) VALUES (1, 'Alice')")
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
                conn.exec_driver_sql("INSERT INTO readonly_table (id, name) VALUES (2, 'Bob')")
    finally:
        engine.dispose()


def test_sql_database_resource_is_pickleable(tmp_path) -> None:
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


def test_worker_sql_config_can_resolve_connection_from_env(monkeypatch, tmp_path) -> None:
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
        payload.model_copy(update={"connection_env_var": "BOTI_WORKER_DSN"})
    )
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "CREATE TABLE env_worker_table (id INTEGER PRIMARY KEY, name TEXT)"
            )
            conn.exec_driver_sql("INSERT INTO env_worker_table (id, name) VALUES (1, 'Alice')")
        with engine.connect() as conn:
            assert conn.execute(text("SELECT name FROM env_worker_table")).scalar() == "Alice"
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_query_only_false_disables_async_session_guard() -> None:
    """Verify async query-only mode can be explicitly disabled."""
    config = SqlDatabaseConfig(
        connection_url="mysql+asyncmy://user:pass@localhost/test_db",
        query_only=False,
    )

    async with AsyncSqlDatabaseResource(config) as db:
        session = db.session()

        with pytest.raises(UnmappedInstanceError):
            session.add(object())


@pytest.mark.asyncio
async def test_async_resource_aclose_is_idempotent() -> None:
    config = SqlDatabaseConfig(
        connection_url="mysql+asyncmy://user:pass@localhost/test_db", query_only=False
    )

    resource = AsyncSqlDatabaseResource(config)
    await resource.__aenter__()
    assert resource.closed is False

    await resource.aclose()
    assert resource.closed is True

    # Repeated closes should be no-ops.
    await resource.aclose()
    assert resource.closed is True


@pytest.mark.asyncio
async def test_async_resource_aclose_is_safe_under_concurrency(monkeypatch) -> None:
    config = SqlDatabaseConfig(
        connection_url="mysql+asyncmy://user:pass@localhost/test_db", query_only=False
    )
    resource = AsyncSqlDatabaseResource(config)

    class _DummyEngine:
        sync_engine = object()

    async def fake_get_or_create_async(*_args, **_kwargs) -> tuple[_DummyEngine, bool]:
        return _DummyEngine(), False

    release_calls: list[tuple] = []

    async def fake_release_async(key, logger=None) -> None:
        release_calls.append((key, logger))

    monkeypatch.setattr(EngineRegistry, "get_or_create_async", fake_get_or_create_async)
    monkeypatch.setattr(EngineRegistry, "release_async", fake_release_async)
    monkeypatch.setattr("boti_data.db.sql_resource.ensure_greenlet_available", lambda: None)
    monkeypatch.setattr(
        "boti_data.db.sql_resource._validate_query_only_support", lambda _parsed: None
    )

    await resource.__aenter__()
    await asyncio.gather(resource.aclose(), resource.aclose(), resource.aclose())

    assert resource.closed is True
    assert len(release_calls) == 1
    assert resource._engine is None


async def _prepare_race_test_resource(
    monkeypatch,
) -> tuple[AsyncSqlDatabaseResource, tuple]:
    """Build an entered AsyncSqlDatabaseResource whose engine is registered under
    a shared, refcounted key with a slow (real-suspension) dispose(), then bump
    the refcount to 2 to simulate a sibling resource holding the same engine."""
    config = SqlDatabaseConfig(
        connection_url="mysql+asyncmy://user:pass@localhost/test_db", query_only=False
    )
    resource = AsyncSqlDatabaseResource(config)

    engine_key = ("probe-shared-key",)
    shared_engine = _SlowDisposeEngine()

    monkeypatch.setattr(
        EngineRegistry,
        "get_or_create_async",
        _make_fake_get_or_create_async(engine_key, shared_engine),
    )
    monkeypatch.setattr(resource, "_get_engine_key", lambda: engine_key)
    monkeypatch.setattr("boti_data.db.sql_resource.ensure_greenlet_available", lambda: None)
    monkeypatch.setattr(
        "boti_data.db.sql_resource._validate_query_only_support", lambda _parsed: None
    )

    await resource.__aenter__()

    # Simulate a sibling resource independently holding a reference to the
    # same engine: bump ref_count from 1 (this resource's own) to 2.
    with EngineRegistry._lock:
        EngineRegistry._registry[engine_key]["ref_count"] = 2

    return resource, engine_key


@pytest.mark.asyncio
async def test_async_resource_aclose_race_does_not_leak_shared_engine(monkeypatch) -> None:
    """Regression: aclose() used to check state, await release_async(), and
    only afterward reset self._engine — two concurrent aclose() calls could
    both pass the check before either reset state, double-releasing a shared
    engine (ref_count > 1) while a sibling resource still held a live
    reference. Proven here through AsyncSqlDatabaseResource's public API,
    using LifecycleCore's aclose() barrier instead of a hand-rolled check.
    """
    resource, engine_key = await _prepare_race_test_resource(monkeypatch)

    # Race this resource's own aclose() against itself.
    await asyncio.gather(resource.aclose(), resource.aclose())

    assert _SlowDisposeEngine.dispose_calls == 0, (
        "engine was disposed while the simulated sibling still holds a reference"
    )
    remaining = EngineRegistry._registry.get(engine_key)
    assert remaining is not None and remaining["ref_count"] == 1, (
        "this resource's aclose() should decrement the shared refcount by "
        "exactly 1 (one logical release), not more"
    )

    # Cleanup: release the simulated sibling's reference too.
    with EngineRegistry._lock:
        EngineRegistry._registry.pop(engine_key, None)
    assert resource._session_factory is None
