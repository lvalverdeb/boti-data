"""
Tests for SqlRepository/AsyncSqlRepository — single-row get/insert/update/delete
by primary key, returning plain dicts — and SqlUnitOfWork/AsyncSqlUnitOfWork,
which compose several repositories into one atomic transaction.
"""

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.db.sql_repository import (
    AsyncSqlRepository,
    AsyncSqlUnitOfWork,
    SqlRepository,
    SqlUnitOfWork,
    _require_unique_key_for_clickhouse,
)
from boti_data.db.sql_resource import AsyncSqlDatabaseResource, SqlDatabaseResource


def _create_sync_table(engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")


def _create_two_sync_tables(engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE audit_log (id INTEGER PRIMARY KEY, msg TEXT)")
        conn.exec_driver_sql("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, val TEXT)")


async def _create_two_async_tables(engine) -> None:
    metadata = MetaData()
    Table("audit_log", metadata, Column("id", Integer, primary_key=True), Column("msg", String))
    Table("embeddings", metadata, Column("id", Integer, primary_key=True), Column("val", String))
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


def test_sync_repository_crud_round_trip(tmp_path) -> None:
    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{tmp_path / 'crud.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    with SqlDatabaseResource(config) as setup:
        _create_sync_table(setup.engine)

        with SqlRepository(config, "users", query_only=False) as repo:
            assert repo.engine is setup.engine

            inserted = repo.insert({"id": 1, "name": "Ada"})
            assert inserted == {"id": 1, "name": "Ada"}

            got = repo.get(1)
            assert got == {"id": 1, "name": "Ada"}

            updated = repo.update(1, {"name": "Grace"})
            assert updated == {"id": 1, "name": "Grace"}

            deleted = repo.delete(1)
            assert deleted is True


def test_sync_repository_missing_pk_returns_none_or_false(tmp_path) -> None:
    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{tmp_path / 'missing_pk.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    with SqlDatabaseResource(config) as setup:
        _create_sync_table(setup.engine)

        # query_only defaults to True on SqlRepository regardless of the
        # config's own value — but a plain (non "mode=ro&uri=true") SQLite
        # file DSN can't satisfy native read-only enforcement at all, so this
        # still needs query_only=False even though no write actually happens
        # below (get/update/delete on a missing pk never reach session.commit()).
        with SqlRepository(config, "users", query_only=False) as repo:
            assert repo.get(999) is None
            assert repo.update(999, {"name": "x"}) is None
            assert repo.delete(999) is False


def test_sync_repository_query_only_rejects_writes() -> None:
    def create_db(path) -> None:
        engine = create_engine(f"sqlite:///{path}")
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        engine.dispose()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "readonly.db"
        create_db(db_path)
        read_only_url = f"sqlite:///file:{db_path}?mode=ro&uri=true"
        config = SqlDatabaseConfig(
            connection_url=read_only_url, poolclass="sqlalchemy.pool.NullPool"
        )

        with SqlRepository(config, "users") as repo:
            assert repo.get(1) == {"id": 1, "name": "Alice"}

            with pytest.raises(SQLAlchemyError, match="read-only"):
                repo.insert({"id": 2, "name": "Bob"})

            with pytest.raises(SQLAlchemyError, match="read-only"):
                repo.update(1, {"name": "Bob"})

            with pytest.raises(SQLAlchemyError, match="read-only"):
                repo.delete(1)


def test_sync_repository_query_only_default_overrides_permissive_config(tmp_path) -> None:
    """A config with query_only=False must not silently grant SqlRepository
    write access — the repository's own query_only=True default always wins
    unless a caller opts out explicitly, even when the config already says
    otherwise (e.g. a config object reused from a writable context)."""
    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{tmp_path / 'override.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    with SqlDatabaseResource(config) as setup:
        _create_sync_table(setup.engine)

    with pytest.raises(SQLAlchemyError, match="native read-only"):
        SqlRepository(config, "users")


async def _create_async_table(engine) -> None:
    metadata = MetaData()
    Table("users", metadata, Column("id", Integer, primary_key=True), Column("name", String))
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


@pytest.mark.asyncio
async def test_async_repository_crud_round_trip(tmp_path) -> None:
    config = SqlDatabaseConfig(
        connection_url=f"sqlite+aiosqlite:///{tmp_path / 'async_crud.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    async with AsyncSqlDatabaseResource(config) as setup:
        await _create_async_table(setup.engine)

        async with AsyncSqlRepository(config, "users", query_only=False) as repo:
            assert repo.engine is setup.engine

            inserted = await repo.insert({"id": 1, "name": "Ada"})
            assert inserted == {"id": 1, "name": "Ada"}

            got = await repo.get(1)
            assert got == {"id": 1, "name": "Ada"}

            updated = await repo.update(1, {"name": "Grace"})
            assert updated == {"id": 1, "name": "Grace"}

            deleted = await repo.delete(1)
            assert deleted is True


@pytest.mark.asyncio
async def test_async_repository_missing_pk_returns_none_or_false(tmp_path) -> None:
    config = SqlDatabaseConfig(
        connection_url=f"sqlite+aiosqlite:///{tmp_path / 'async_missing_pk.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    async with AsyncSqlDatabaseResource(config) as setup:
        await _create_async_table(setup.engine)

        # See the sync equivalent's comment: query_only defaults to True on
        # AsyncSqlRepository too, and a plain SQLite file DSN can't satisfy
        # native read-only enforcement, so this needs query_only=False even
        # though no write actually happens for a missing pk.
        async with AsyncSqlRepository(config, "users", query_only=False) as repo:
            assert await repo.get(999) is None
            assert await repo.update(999, {"name": "x"}) is None
            assert await repo.delete(999) is False


@pytest.mark.asyncio
async def test_async_repository_query_only_rejects_writes() -> None:
    # SQLite in-memory DBs cannot enforce read-only mode (query_only=True
    # raises at construction time for those), so — mirroring the sync
    # equivalent in test_sql_manager_lifecycle.py — populate a real file-based
    # SQLite DB first, then reopen it read-only via 'mode=ro&uri=true'.
    def create_db(path) -> None:
        engine = create_engine(f"sqlite:///{path}")
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        engine.dispose()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "readonly.db"
        create_db(db_path)
        read_only_url = f"sqlite+aiosqlite:///file:{db_path}?mode=ro&uri=true"
        config = SqlDatabaseConfig(
            connection_url=read_only_url, poolclass="sqlalchemy.pool.NullPool"
        )

        async with AsyncSqlRepository(config, "users") as repo:
            assert await repo.get(1) == {"id": 1, "name": "Alice"}

            with pytest.raises(SQLAlchemyError, match="read-only"):
                await repo.insert({"id": 2, "name": "Bob"})

            with pytest.raises(SQLAlchemyError, match="read-only"):
                await repo.update(1, {"name": "Bob"})

            with pytest.raises(SQLAlchemyError, match="read-only"):
                await repo.delete(1)


@pytest.mark.asyncio
async def test_async_repository_query_only_default_overrides_permissive_config(tmp_path) -> None:
    """Async equivalent of the sync override test above."""
    config = SqlDatabaseConfig(
        connection_url=f"sqlite+aiosqlite:///{tmp_path / 'async_override.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    async with AsyncSqlDatabaseResource(config) as setup:
        await _create_async_table(setup.engine)

    with pytest.raises(SQLAlchemyError, match="native read-only"):
        async with AsyncSqlRepository(config, "users"):
            pass


def test_sync_repository_requires_exactly_one_of_config_or_session(tmp_path) -> None:
    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{tmp_path / 'exclusive.db'}", query_only=False
    )
    with SqlDatabaseResource(config) as setup:
        _create_sync_table(setup.engine)
        with setup.session() as session:
            with pytest.raises(ValueError, match="exactly one"):
                SqlRepository(config, "users", session=session)

    with pytest.raises(ValueError, match="exactly one"):
        SqlRepository(None, "users")


def test_sync_repositories_sharing_an_injected_session_are_atomic(tmp_path) -> None:
    """Two SqlRepository instances built from one caller-supplied session,
    against two different tables, must share one transaction: rolling back
    the session must undo both repositories' writes together."""
    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{tmp_path / 'shared_session.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    with SqlDatabaseResource(config) as setup:
        _create_two_sync_tables(setup.engine)

        with setup.session() as session:
            with SqlRepository(session=session, table_name="audit_log") as audit:
                with SqlRepository(session=session, table_name="embeddings") as embeddings:
                    audit.insert({"id": 1, "msg": "enroll"})
                    embeddings.insert({"id": 1, "val": "x"})
                    session.rollback()

        with setup.session() as session:
            assert session.execute(text("SELECT * FROM audit_log")).fetchall() == []
            assert session.execute(text("SELECT * FROM embeddings")).fetchall() == []


def test_sql_unit_of_work_commits_multiple_tables_together(tmp_path) -> None:
    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{tmp_path / 'uow_commit.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    with SqlDatabaseResource(config) as setup:
        _create_two_sync_tables(setup.engine)

        with SqlUnitOfWork(config, query_only=False) as uow:
            uow.repository("audit_log").insert({"id": 1, "msg": "enroll"})
            uow.repository("embeddings").insert({"id": 1, "val": "x"})

        with setup.session() as session:
            assert session.execute(text("SELECT * FROM audit_log")).fetchall() == [(1, "enroll")]
            assert session.execute(text("SELECT * FROM embeddings")).fetchall() == [(1, "x")]


def test_sql_unit_of_work_rolls_back_multiple_tables_together(tmp_path) -> None:
    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{tmp_path / 'uow_rollback.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    with SqlDatabaseResource(config) as setup:
        _create_two_sync_tables(setup.engine)

        with pytest.raises(RuntimeError, match="boom"):
            with SqlUnitOfWork(config, query_only=False) as uow:
                uow.repository("audit_log").insert({"id": 1, "msg": "enroll"})
                uow.repository("embeddings").insert({"id": 1, "val": "x"})
                raise RuntimeError("boom")

        with setup.session() as session:
            assert session.execute(text("SELECT * FROM audit_log")).fetchall() == []
            assert session.execute(text("SELECT * FROM embeddings")).fetchall() == []


def test_sql_unit_of_work_query_only_defaults_true(tmp_path) -> None:
    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{tmp_path / 'uow_default_ro.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    with SqlDatabaseResource(config) as setup:
        _create_sync_table(setup.engine)

    with pytest.raises(SQLAlchemyError, match="native read-only"):
        SqlUnitOfWork(config)


def test_sql_unit_of_work_read_only_exits_cleanly_with_no_writes() -> None:
    """Regression: __exit__ used to call session.commit() unconditionally on
    a clean exit, and ReadOnlySession.commit() always raises regardless of
    whether anything was actually written — so a pure-read query_only=True
    (the default) unit of work could never exit cleanly."""

    def create_db(path) -> None:
        engine = create_engine(f"sqlite:///{path}")
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        engine.dispose()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "readonly.db"
        create_db(db_path)
        read_only_url = f"sqlite:///file:{db_path}?mode=ro&uri=true"
        config = SqlDatabaseConfig(
            connection_url=read_only_url, poolclass="sqlalchemy.pool.NullPool"
        )

        with SqlUnitOfWork(config) as uow:
            users = uow.repository("users")
            assert users.get(1) == {"id": 1, "name": "Alice"}
        # No exception on __exit__ is the actual regression test.


def test_sync_repository_injected_session_ignores_its_own_query_only(tmp_path) -> None:
    """A session's own class (ReadOnlySession or not) governs writability when
    a session is injected — the repository's query_only parameter is inert
    there, since this path never constructs a new guarded session."""

    def create_db(path) -> None:
        engine = create_engine(f"sqlite:///{path}")
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        engine.dispose()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "readonly.db"
        create_db(db_path)
        read_only_url = f"sqlite:///file:{db_path}?mode=ro&uri=true"
        config = SqlDatabaseConfig(
            connection_url=read_only_url, poolclass="sqlalchemy.pool.NullPool"
        )

        with SqlDatabaseResource(config) as db:
            with db.session() as session:
                assert type(session).__name__ == "ReadOnlySession"
                # query_only=False here must NOT grant writes — the injected
                # session is already read-only-guarded regardless.
                with SqlRepository(session=session, table_name="users", query_only=False) as repo:
                    assert repo.get(1) == {"id": 1, "name": "Alice"}
                    with pytest.raises(SQLAlchemyError, match="read-only"):
                        repo.insert({"id": 2, "name": "Bob"})


@pytest.mark.asyncio
async def test_async_repositories_sharing_an_injected_session_are_atomic(tmp_path) -> None:
    config = SqlDatabaseConfig(
        connection_url=f"sqlite+aiosqlite:///{tmp_path / 'async_shared_session.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    async with AsyncSqlDatabaseResource(config) as setup:
        await _create_two_async_tables(setup.engine)

        async with setup.session() as session:
            async with AsyncSqlRepository(session=session, table_name="audit_log") as audit:
                async with AsyncSqlRepository(
                    session=session, table_name="embeddings"
                ) as embeddings:
                    await audit.insert({"id": 1, "msg": "enroll"})
                    await embeddings.insert({"id": 1, "val": "x"})
                    await session.rollback()

        async with setup.session() as session:
            assert (await session.execute(text("SELECT * FROM audit_log"))).fetchall() == []
            assert (await session.execute(text("SELECT * FROM embeddings"))).fetchall() == []


@pytest.mark.asyncio
async def test_async_sql_unit_of_work_commits_multiple_tables_together(tmp_path) -> None:
    config = SqlDatabaseConfig(
        connection_url=f"sqlite+aiosqlite:///{tmp_path / 'async_uow_commit.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    async with AsyncSqlDatabaseResource(config) as setup:
        await _create_two_async_tables(setup.engine)

        async with AsyncSqlUnitOfWork(config, query_only=False) as uow:
            audit = await uow.repository("audit_log")
            embeddings = await uow.repository("embeddings")
            await audit.insert({"id": 1, "msg": "enroll"})
            await embeddings.insert({"id": 1, "val": "x"})

        async with setup.session() as session:
            audit_rows = (await session.execute(text("SELECT * FROM audit_log"))).fetchall()
            embeddings_rows = (await session.execute(text("SELECT * FROM embeddings"))).fetchall()
            assert audit_rows == [(1, "enroll")]
            assert embeddings_rows == [(1, "x")]


@pytest.mark.asyncio
async def test_async_sql_unit_of_work_rolls_back_multiple_tables_together(tmp_path) -> None:
    config = SqlDatabaseConfig(
        connection_url=f"sqlite+aiosqlite:///{tmp_path / 'async_uow_rollback.db'}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    async with AsyncSqlDatabaseResource(config) as setup:
        await _create_two_async_tables(setup.engine)

        with pytest.raises(RuntimeError, match="boom"):
            async with AsyncSqlUnitOfWork(config, query_only=False) as uow:
                audit = await uow.repository("audit_log")
                embeddings = await uow.repository("embeddings")
                await audit.insert({"id": 1, "msg": "enroll"})
                await embeddings.insert({"id": 1, "val": "x"})
                raise RuntimeError("boom")

        async with setup.session() as session:
            assert (await session.execute(text("SELECT * FROM audit_log"))).fetchall() == []
            assert (await session.execute(text("SELECT * FROM embeddings"))).fetchall() == []


@pytest.mark.asyncio
async def test_async_sql_unit_of_work_read_only_exits_cleanly_with_no_writes() -> None:
    """Async equivalent of the sync regression test above."""

    def create_db(path) -> None:
        engine = create_engine(f"sqlite:///{path}")
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        engine.dispose()

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "readonly.db"
        create_db(db_path)
        read_only_url = f"sqlite+aiosqlite:///file:{db_path}?mode=ro&uri=true"
        config = SqlDatabaseConfig(
            connection_url=read_only_url, poolclass="sqlalchemy.pool.NullPool"
        )

        async with AsyncSqlUnitOfWork(config) as uow:
            users = await uow.repository("users")
            assert await users.get(1) == {"id": 1, "name": "Alice"}
        # No exception on __aexit__ is the actual regression test.


def test_repository_dependency_free_import() -> None:
    """Importing SqlRepository must not pull in dask/pandas/polars."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from boti_data.db.sql_repository import SqlRepository, AsyncSqlRepository\n"
            "heavy = {'dask', 'pandas', 'polars'} & set(sys.modules)\n"
            "assert not heavy, f'unexpectedly imported: {heavy}'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_sql_unit_of_work_rejects_clickhouse() -> None:
    """ClickHouse has no real cross-table transactional guarantees, so
    SqlUnitOfWork must reject it instead of silently offering an atomicity
    it can't back."""
    config = SqlDatabaseConfig(connection_url="clickhousedb://user:pass@localhost:8123/test_db")

    with pytest.raises(SQLAlchemyError, match="does not support ClickHouse"):
        SqlUnitOfWork(config)


@pytest.mark.asyncio
async def test_async_sql_unit_of_work_rejects_clickhouse() -> None:
    """Async counterpart of test_sql_unit_of_work_rejects_clickhouse."""
    config = SqlDatabaseConfig(connection_url="clickhousedb://user:pass@localhost:8123/test_db")

    with pytest.raises(SQLAlchemyError, match="does not support ClickHouse"):
        AsyncSqlUnitOfWork(config)


def test_require_unique_key_for_clickhouse_blocks_without_assertion() -> None:
    """ClickHouse's ORDER BY/primary key is a sort prefix, not a uniqueness
    constraint, so get()/update()/delete() must refuse to run against it
    unless the caller explicitly asserts the key is actually unique.
    create_engine() is lazy (no connection dialed), so this is hermetic."""
    engine = create_engine("clickhousedb://user:pass@localhost:8123/test_db")

    with pytest.raises(SQLAlchemyError, match="assume_unique_key"):
        _require_unique_key_for_clickhouse(engine, assume_unique_key=False, operation="x")

    _require_unique_key_for_clickhouse(engine, assume_unique_key=True, operation="x")


def test_require_unique_key_for_clickhouse_ignores_other_backends() -> None:
    """Non-ClickHouse backends have real primary-key guarantees, so the guard
    must never fire for them regardless of assume_unique_key."""
    engine = create_engine("sqlite://")

    _require_unique_key_for_clickhouse(engine, assume_unique_key=False, operation="x")


def test_duckdb_repository_crud_round_trip(tmp_path) -> None:
    """DuckDB is embedded, so — unlike ClickHouse — this is a genuine hermetic
    round-trip test, no external server needed."""
    config = SqlDatabaseConfig(
        connection_url=f"duckdb:///{tmp_path / 'crud.duckdb'}",
        query_only=False,
    )

    with SqlDatabaseResource(config) as setup:
        with setup.engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE users (id BIGINT PRIMARY KEY, name TEXT)")

        with SqlRepository(config, "users", query_only=False) as repo:
            inserted = repo.insert({"id": 1, "name": "Ada"})
            assert inserted == {"id": 1, "name": "Ada"}

            got = repo.get(1)
            assert got == {"id": 1, "name": "Ada"}

            updated = repo.update(1, {"name": "Grace"})
            assert updated == {"id": 1, "name": "Grace"}

            deleted = repo.delete(1)
            assert deleted is True


def test_duckdb_repository_composite_key_round_trip(tmp_path) -> None:
    """Regression test for a real bug found during implementation: duckdb-engine
    doesn't reflect PRIMARY KEY/UNIQUE constraints at all, so without
    _recover_duckdb_primary_key (sql_model_registry.py) every DuckDB table
    looked keyless and SqlRepository silently bound to an arbitrary first
    column instead of the real composite key — get()/delete() on 'electronics'
    touched an arbitrary row / every row sharing that category instead of the
    one row identified by the full (category, order_id) key."""
    config = SqlDatabaseConfig(
        connection_url=f"duckdb:///{tmp_path / 'composite.duckdb'}",
        query_only=False,
    )

    with SqlDatabaseResource(config) as setup:
        with setup.engine.begin() as conn:
            conn.exec_driver_sql(
                "CREATE TABLE orders ("
                "category VARCHAR NOT NULL, order_id BIGINT NOT NULL, amount DOUBLE, "
                "PRIMARY KEY (category, order_id))"
            )
            conn.exec_driver_sql(
                "INSERT INTO orders VALUES "
                "('electronics', 1, 100.0), ('electronics', 2, 200.0), "
                "('electronics', 3, 300.0), ('books', 4, 15.0)"
            )

        with SqlRepository(config, "orders", query_only=False) as repo:
            assert repo.get(("electronics", 1)) == {
                "category": "electronics",
                "order_id": 1,
                "amount": 100.0,
            }
            assert repo.get(("electronics", 2)) == {
                "category": "electronics",
                "order_id": 2,
                "amount": 200.0,
            }

            assert repo.delete(("electronics", 1)) is True

        with setup.session() as session:
            remaining = session.execute(
                text("SELECT category, order_id FROM orders ORDER BY order_id")
            ).fetchall()
            assert remaining == [("electronics", 2), ("electronics", 3), ("books", 4)]


def test_duckdb_query_only_enforced_server_side(tmp_path) -> None:
    """Verify query_only=True is enforced by DuckDB itself (a connect-time
    read_only flag applied via connect_args), not just by boti-data's own
    ReadOnlySession guard — bypass the ORM entirely and write straight through
    the raw engine connection."""
    db_path = tmp_path / "readonly.duckdb"
    setup_config = SqlDatabaseConfig(connection_url=f"duckdb:///{db_path}", query_only=False)
    with SqlDatabaseResource(setup_config) as setup:
        with setup.engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE t (id BIGINT PRIMARY KEY)")

    ro_config = SqlDatabaseConfig(connection_url=f"duckdb:///{db_path}", query_only=True)
    with SqlDatabaseResource(ro_config) as ro:
        with ro.engine.connect() as conn:
            assert conn.exec_driver_sql("SELECT count(*) FROM t").scalar() == 0
            with pytest.raises(SQLAlchemyError, match="read-only mode"):
                conn.exec_driver_sql("INSERT INTO t VALUES (1)")


def test_duckdb_sql_unit_of_work_commits_multiple_tables_together(tmp_path) -> None:
    """DuckDB has real multi-table ACID transactions (unlike ClickHouse), so
    SqlUnitOfWork is genuinely supported for it — no rejection guard."""
    config = SqlDatabaseConfig(
        connection_url=f"duckdb:///{tmp_path / 'uow_commit.duckdb'}",
        query_only=False,
    )
    with SqlDatabaseResource(config) as setup:
        with setup.engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE accounts (id BIGINT PRIMARY KEY, balance DOUBLE)")
            conn.exec_driver_sql("CREATE TABLE ledger (id BIGINT PRIMARY KEY, note TEXT)")
            conn.exec_driver_sql("INSERT INTO accounts VALUES (1, 100.0)")

        with SqlUnitOfWork(config, query_only=False) as uow:
            uow.repository("accounts").update(1, {"balance": 50.0})
            uow.repository("ledger").insert({"id": 1, "note": "withdrawal"})

        with setup.session() as session:
            assert session.execute(text("SELECT balance FROM accounts")).scalar() == 50.0
            assert session.execute(text("SELECT note FROM ledger")).scalar() == "withdrawal"


def test_duckdb_sql_unit_of_work_rolls_back_multiple_tables_together(tmp_path) -> None:
    """Prove the rollback is real ACID behavior, not just 'no exception on
    construction': trigger an actual PRIMARY KEY constraint violation
    mid-transaction and confirm both tables revert to their pre-transaction
    state, not just the table the violation happened on."""
    config = SqlDatabaseConfig(
        connection_url=f"duckdb:///{tmp_path / 'uow_rollback.duckdb'}",
        query_only=False,
    )
    with SqlDatabaseResource(config) as setup:
        with setup.engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE accounts (id BIGINT PRIMARY KEY, balance DOUBLE)")
            conn.exec_driver_sql("CREATE TABLE ledger (id BIGINT PRIMARY KEY, note TEXT)")
            conn.exec_driver_sql("INSERT INTO accounts VALUES (1, 100.0)")

        with pytest.raises(SQLAlchemyError, match="[Dd]uplicate"):
            with SqlUnitOfWork(config, query_only=False) as uow:
                uow.repository("accounts").update(1, {"balance": 50.0})
                uow.repository("ledger").insert({"id": 1, "note": "withdrawal"})
                uow.repository("ledger").insert({"id": 1, "note": "duplicate-pk-should-fail"})

        with setup.session() as session:
            assert session.execute(text("SELECT balance FROM accounts")).scalar() == 100.0
            assert session.execute(text("SELECT * FROM ledger")).fetchall() == []
