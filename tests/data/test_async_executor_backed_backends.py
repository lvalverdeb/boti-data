"""Async parity for the sync-only dialects: DuckDB and ClickHouse (PRD R6).

Each test here mirrors one sync test for the same backend (named in its
docstring), so the async and sync counts per backend stay equal. DuckDB is
embedded, so its tests are real round trips on a temp file; ClickHouse needs a
server, so its DBAPI is replaced at the clickhouse_connect boundary.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from boti_data import (
    AsyncSqlDatabaseResource,
    AsyncSqlRepository,
    AsyncSqlUnitOfWork,
    SqlDatabaseConfig,
)
from boti_data.db.sql_engine import _validate_async_driver_url
from boti_data.db.sql_readonly import ReadOnlySession, is_read_only_session
from boti_data.db.threaded_async import ThreadedAsyncEngine, ThreadedAsyncSession

pytestmark = pytest.mark.asyncio


def _duckdb(tmp_path, name: str, *, query_only: bool) -> SqlDatabaseConfig:
    return SqlDatabaseConfig(connection_url=f"duckdb:///{tmp_path / name}", query_only=query_only)


async def _create_users(config: SqlDatabaseConfig) -> None:
    async with AsyncSqlDatabaseResource(config) as setup:
        async with setup.engine.begin() as conn:
            await conn.exec_driver_sql("CREATE TABLE users (id BIGINT PRIMARY KEY, name TEXT)")


# ---------------------------------------------------------------- DuckDB ---


async def test_async_resource_accepts_duckdb_dsn(tmp_path) -> None:
    """Mirror of test_sync_resource_accepts_duckdb_dsn."""
    config = _duckdb(tmp_path, "accept.duckdb", query_only=False)
    async with AsyncSqlDatabaseResource(config) as db:
        assert db.engine.url.get_backend_name() == "duckdb"
        assert isinstance(db.engine, ThreadedAsyncEngine)
        assert isinstance(db.session(), ThreadedAsyncSession)


async def test_async_validate_driver_url_accepts_duckdb(tmp_path) -> None:
    """Mirror of test_validate_query_only_support_accepts_duckdb: the async
    validator no longer rejects the sync-only dialect."""
    parsed = _validate_async_driver_url(f"duckdb:///{tmp_path / 'v.duckdb'}")
    assert parsed.get_backend_name() == "duckdb"


async def test_async_duckdb_query_only_enforced_server_side(tmp_path) -> None:
    """Mirror of test_duckdb_query_only_enforced_server_side (and of
    test_build_sync_engine_kwargs_injects_duckdb_read_only: the facade is built
    with the sync kwargs, so the connect-time read_only flag applies)."""
    await _create_users(_duckdb(tmp_path, "ro.duckdb", query_only=False))
    async with AsyncSqlDatabaseResource(_duckdb(tmp_path, "ro.duckdb", query_only=True)) as ro:
        assert ro.engine.write_lock is None
        async with ro.engine.connect() as conn:
            assert await conn.scalar(text("SELECT count(*) FROM users")) == 0
            with pytest.raises(SQLAlchemyError, match="read-only mode"):
                await conn.exec_driver_sql("INSERT INTO users VALUES (1, 'x')")


async def test_async_duckdb_repository_crud_round_trip(tmp_path) -> None:
    """Mirror of test_duckdb_repository_crud_round_trip."""
    config = _duckdb(tmp_path, "crud.duckdb", query_only=False)
    await _create_users(config)
    async with AsyncSqlRepository(config, "users", query_only=False) as repo:
        assert await repo.insert({"id": 1, "name": "Ada"}) == {"id": 1, "name": "Ada"}
        assert await repo.get(1) == {"id": 1, "name": "Ada"}
        assert await repo.update(1, {"name": "Grace"}) == {"id": 1, "name": "Grace"}
        assert await repo.delete(1) is True


async def test_async_duckdb_repository_composite_key_round_trip(tmp_path) -> None:
    """Mirror of test_duckdb_repository_composite_key_round_trip."""
    config = _duckdb(tmp_path, "composite.duckdb", query_only=False)
    async with AsyncSqlDatabaseResource(config) as setup:
        async with setup.engine.begin() as conn:
            await conn.exec_driver_sql(
                "CREATE TABLE orders (category VARCHAR NOT NULL, order_id BIGINT NOT NULL, "
                "amount DOUBLE, PRIMARY KEY (category, order_id))"
            )
            await conn.exec_driver_sql(
                "INSERT INTO orders VALUES ('electronics', 1, 100.0), ('electronics', 2, 200.0), "
                "('books', 4, 15.0)"
            )
        async with AsyncSqlRepository(config, "orders", query_only=False) as repo:
            assert await repo.get(("electronics", 1)) == {
                "category": "electronics",
                "order_id": 1,
                "amount": 100.0,
            }
            assert await repo.delete(("electronics", 1)) is True
        async with setup.session() as session:
            rows = await session.execute(
                text("SELECT category, order_id FROM orders ORDER BY order_id")
            )
            assert rows.fetchall() == [("electronics", 2), ("books", 4)]


async def test_async_duckdb_unit_of_work_commits_multiple_tables_together(tmp_path) -> None:
    """Mirror of test_duckdb_sql_unit_of_work_commits_multiple_tables_together."""
    config = _duckdb(tmp_path, "uow_commit.duckdb", query_only=False)
    async with AsyncSqlDatabaseResource(config) as setup:
        async with setup.engine.begin() as conn:
            await conn.exec_driver_sql(
                "CREATE TABLE accounts (id BIGINT PRIMARY KEY, balance DOUBLE)"
            )
            await conn.exec_driver_sql("CREATE TABLE ledger (id BIGINT PRIMARY KEY, note TEXT)")
            await conn.exec_driver_sql("INSERT INTO accounts VALUES (1, 100.0)")
        async with AsyncSqlUnitOfWork(config, query_only=False) as uow:
            await (await uow.repository("accounts")).update(1, {"balance": 50.0})
            await (await uow.repository("ledger")).insert({"id": 1, "note": "withdrawal"})
        async with setup.session() as session:
            assert await session.scalar(text("SELECT balance FROM accounts")) == 50.0
            assert await session.scalar(text("SELECT note FROM ledger")) == "withdrawal"


async def test_async_duckdb_unit_of_work_rolls_back_multiple_tables_together(tmp_path) -> None:
    """Mirror of test_duckdb_sql_unit_of_work_rolls_back_multiple_tables_together."""
    config = _duckdb(tmp_path, "uow_rollback.duckdb", query_only=False)
    async with AsyncSqlDatabaseResource(config) as setup:
        async with setup.engine.begin() as conn:
            await conn.exec_driver_sql(
                "CREATE TABLE accounts (id BIGINT PRIMARY KEY, balance DOUBLE)"
            )
            await conn.exec_driver_sql("CREATE TABLE ledger (id BIGINT PRIMARY KEY, note TEXT)")
            await conn.exec_driver_sql("INSERT INTO accounts VALUES (1, 100.0)")
        with pytest.raises(SQLAlchemyError, match="[Dd]uplicate"):
            async with AsyncSqlUnitOfWork(config, query_only=False) as uow:
                await (await uow.repository("accounts")).update(1, {"balance": 50.0})
                ledger = await uow.repository("ledger")
                await ledger.insert({"id": 1, "note": "withdrawal"})
                await ledger.insert({"id": 1, "note": "duplicate-pk-should-fail"})
        async with setup.session() as session:
            assert await session.scalar(text("SELECT balance FROM accounts")) == 100.0
            assert (await session.execute(text("SELECT * FROM ledger"))).fetchall() == []


async def test_async_duckdb_read_only_unit_of_work_exits_without_commit(tmp_path) -> None:
    """Mirror of the read-only session guard: a query_only unit of work over the
    threaded session is recognised as read-only and never commits."""
    config = _duckdb(tmp_path, "ro_uow.duckdb", query_only=False)
    await _create_users(config)
    async with AsyncSqlUnitOfWork(_duckdb(tmp_path, "ro_uow.duckdb", query_only=True)) as uow:
        assert is_read_only_session(uow._session)
        assert isinstance(uow._session.sync_session, ReadOnlySession)
        assert await (await uow.repository("users")).get(1) is None


async def test_async_duckdb_serialises_writers_and_runs_reads_concurrently(tmp_path) -> None:
    """A read-write DuckDB resource runs its statements one at a time behind a
    lock; a query_only one has no lock and overlaps reads."""
    config = _duckdb(tmp_path, "lock.duckdb", query_only=False)
    await _create_users(config)
    async with AsyncSqlDatabaseResource(config) as rw:
        assert rw.engine.write_lock is not None

        async def insert(i: int) -> None:
            async with rw.engine.begin() as conn:
                await conn.exec_driver_sql(f"INSERT INTO users VALUES ({i}, 'u{i}')")

        await asyncio.gather(*(insert(i) for i in range(20)))
        async with rw.engine.connect() as conn:
            assert await conn.scalar(text("SELECT count(*) FROM users")) == 20
    async with AsyncSqlDatabaseResource(_duckdb(tmp_path, "lock.duckdb", query_only=True)) as ro:

        async def count() -> int:
            async with ro.engine.connect() as conn:
                return await conn.scalar(text("SELECT count(*) FROM users"))

        assert await asyncio.gather(*(count() for _ in range(8))) == [20] * 8


# ------------------------------------------------------------ ClickHouse ---

_CLICKHOUSE_DSN = "clickhousedb://user:pass@localhost:8123/test_db"


class _FakeCursor:
    def __init__(self, log: list[tuple[str, Any]]) -> None:
        self._log = log
        self.description: Any = None
        self.rowcount = -1
        self.arraysize = 1

    def execute(self, sql: str, parameters: Any = None, **_driver_kwargs: Any) -> None:
        self._log.append((sql, parameters))
        self.description = [("one", None, None, None, None, None, None)]

    def executemany(self, sql: str, seq: Any, **_driver_kwargs: Any) -> None:
        self._log.append((sql, seq))

    def fetchone(self) -> Any:
        return (1,)

    def fetchmany(self, size: int | None = None) -> Any:
        return [(1,)]

    def fetchall(self) -> Any:
        return [(1,)]

    def close(self) -> None:
        pass


class _FakeConnection:
    def __init__(self, log: list[tuple[str, Any]]) -> None:
        self._log = log
        self.settings: dict[str, Any] = {}
        self.client = SimpleNamespace(set_client_setting=self.settings.__setitem__)

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._log)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def fake_clickhouse(monkeypatch) -> tuple[list[tuple[str, Any]], list[_FakeConnection]]:
    import clickhouse_connect.dbapi as dbapi

    log: list[tuple[str, Any]] = []
    connections: list[_FakeConnection] = []

    def connect(*_args: Any, **_kwargs: Any) -> _FakeConnection:
        connection = _FakeConnection(log)
        connections.append(connection)
        return connection

    monkeypatch.setattr(dbapi, "connect", connect)
    return log, connections


def _clickhouse(*, query_only: bool) -> SqlDatabaseConfig:
    return SqlDatabaseConfig(
        connection_url=_CLICKHOUSE_DSN, query_only=query_only, poolclass="sqlalchemy.pool.NullPool"
    )


async def test_async_resource_accepts_clickhouse_dsn() -> None:
    """Mirror of test_sync_resource_accepts_clickhouse_dsn."""
    async with AsyncSqlDatabaseResource(_clickhouse(query_only=False)) as db:
        assert db.engine.url.get_backend_name() == "clickhousedb"
        assert isinstance(db.engine, ThreadedAsyncEngine)
        assert db.engine.write_lock is None


async def test_async_validate_driver_url_accepts_clickhouse() -> None:
    """Mirror of test_validate_query_only_support_accepts_clickhouse."""
    assert _validate_async_driver_url(_CLICKHOUSE_DSN).get_backend_name() == "clickhousedb"


async def test_async_clickhouse_round_trip_through_mocked_driver(fake_clickhouse) -> None:
    """Mirror of the sync engine round trip: a statement reaches the DBAPI and
    its rows come back through the threaded facade."""
    log, _ = fake_clickhouse
    async with AsyncSqlDatabaseResource(_clickhouse(query_only=False)) as db:
        async with db.engine.connect() as conn:
            result = await conn.exec_driver_sql("SELECT 1")
            assert result.fetchall() == [(1,)]
    assert any(sql == "SELECT 1" for sql, _ in log)


async def test_async_clickhouse_query_only_sets_readonly_client_setting(fake_clickhouse) -> None:
    """Mirror of _configure_query_only_engine's ClickHouse branch: the connect
    listener applies readonly=1 on every DBAPI connection."""
    _, connections = fake_clickhouse
    async with AsyncSqlDatabaseResource(_clickhouse(query_only=True)) as db:
        async with db.engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
    assert connections and all(c.settings == {"readonly": 1} for c in connections)


async def test_async_clickhouse_query_only_session_is_read_only() -> None:
    """Mirror of the sync ReadOnlySession guard for ClickHouse."""
    async with AsyncSqlDatabaseResource(_clickhouse(query_only=True)) as db:
        session = db.session()
        assert is_read_only_session(session)
        with pytest.raises(SQLAlchemyError, match="read-only"):
            session.add(object())
        await session.close()
