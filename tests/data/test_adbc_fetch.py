"""ADBC partition fetches stay behind the read-only guards and match the SQLAlchemy path."""

from __future__ import annotations

import getpass
import importlib.util
import socket
import sqlite3

import pytest
from sqlalchemy.engine import make_url

from boti_data.db import adbc_fetch
from boti_data.db.partitioned_execution import SqlPartitionExecutor
from boti_data.db.partitioned_types import SqlPartitionSpec
from boti_data.db.sql_config import WorkerSqlConfig

pytest.importorskip("adbc_driver_sqlite")

META = {
    "id": "Int64",
    "name": "string",
    "amount": "Float64",
    "flag": "boolean",
    "ts": "datetime64[ns]",
}
ROWS = [
    (1, "alpha", 1.5, 1, "2026-01-01 10:00:00"),
    (2, None, None, 0, "2026-01-02 11:30:00"),
    (3, "gamma", 3.0, None, None),
    (4, " ", 4.25, 1, "2026-04-04 00:00:00"),
]


@pytest.fixture
def ro_sqlite(tmp_path) -> str:
    path = tmp_path / "ro.sqlite"
    con = sqlite3.connect(path)
    con.execute("create table t (id integer, name text, amount real, flag integer, ts text)")
    con.executemany("insert into t values (?,?,?,?,?)", ROWS)
    con.commit()
    con.close()
    return f"sqlite:///file:{path}?mode=ro&uri=true"


def _config(dsn: str, **overrides) -> WorkerSqlConfig:
    return WorkerSqlConfig(connection_url=dsn, **{"query_only": True, **overrides})


def _fetch(config: WorkerSqlConfig):
    return SqlPartitionExecutor.fetch_partition(
        config=config,
        gate_key="adbc-test",
        max_concurrent_fetches=1,
        partition=SqlPartitionSpec(
            sql="select id, name, amount, flag, ts from t where id >= ? order by id", params=(1,)
        ),
        meta_dtypes=META,
        use_arrow=True,
    )


def test_positional_rewrites_psycopg_pyformat_to_dollar_numbers() -> None:
    sql, params = adbc_fetch.positional(
        "select * from t where a = %(a_1)s and b like '%%x%%' and c in (%(c_1)s, %(c_2)s)",
        {"a_1": 1, "c_1": "p", "c_2": "q"},
        "postgresql",
    )
    assert sql == "select * from t where a = $1 and b like '%x%' and c in ($2, $3)"
    assert params == (1, "p", "q")


def test_positional_leaves_sqlite_qmark_alone() -> None:
    assert adbc_fetch.positional("select 1 where ? = 1", (1,), "sqlite") == (
        "select 1 where ? = 1",
        (1,),
    )


def test_adbc_uri_keeps_sqlite_read_only_and_drops_the_sqlalchemy_driver() -> None:
    assert (
        adbc_fetch.adbc_uri(make_url("sqlite:///file:/tmp/x.db?mode=ro&uri=true"))
        == "file:/tmp/x.db?mode=ro"
    )
    pg = adbc_fetch.adbc_uri(make_url("postgresql+psycopg://u:p@h:5433/db?sslmode=require"))
    assert pg == "postgresql://u:p@h:5433/db?sslmode=require"


def test_eligibility_requires_query_only_a_driver_and_the_flag(ro_sqlite, monkeypatch) -> None:
    assert adbc_fetch.adbc_eligible(_config(ro_sqlite))
    assert not adbc_fetch.adbc_eligible(_config(ro_sqlite, query_only=False))
    assert not adbc_fetch.adbc_eligible(_config(ro_sqlite, adbc=False))
    assert not adbc_fetch.adbc_eligible(_config("mysql+pymysql://u:p@h/db"))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert not adbc_fetch.adbc_eligible(_config(ro_sqlite))


def test_adbc_partition_matches_the_sqlalchemy_partition(ro_sqlite, monkeypatch) -> None:
    calls: list[str] = []
    real = adbc_fetch.fetch_partition_table

    def _spy(config, partition, meta):
        calls.append(partition.sql)
        return real(config, partition, meta)

    monkeypatch.setattr("boti_data.db.partitioned_execution.fetch_partition_table", _spy)
    via_adbc = _fetch(_config(ro_sqlite))
    via_sqlalchemy = _fetch(_config(ro_sqlite, adbc=False))
    assert len(calls) == 1
    assert list(via_adbc.columns) == list(META)
    assert via_adbc.equals(via_sqlalchemy), (via_adbc, via_sqlalchemy)


def test_the_adbc_session_itself_refuses_writes(ro_sqlite) -> None:
    with (
        adbc_fetch.open_read_only(_config(ro_sqlite)) as connection,
        connection.cursor() as cursor,
        pytest.raises(Exception, match="(?i)readonly|read-only|query_only"),
    ):
        cursor.execute("insert into t values (9, 'z', 0.0, 1, null)")


def test_each_fetch_opens_and_closes_its_own_connection(ro_sqlite, monkeypatch) -> None:
    """No connection outlives a fetch: the same rule as the workers' NullPool engines."""
    opened: list[object] = []
    closed: list[object] = []
    real_connect = adbc_fetch._connect

    def _tracking_connect(config):
        connection = real_connect(config)
        opened.append(connection)
        real_close = connection.close
        connection.close = lambda: (closed.append(connection), real_close())
        return connection

    monkeypatch.setattr(adbc_fetch, "_connect", _tracking_connect)
    _fetch(_config(ro_sqlite))
    _fetch(_config(ro_sqlite))
    assert len(opened) == 2 and closed == opened


def test_driver_errors_fall_back_to_sqlalchemy_with_a_warning(
    ro_sqlite, monkeypatch, caplog
) -> None:
    error_type = adbc_fetch.driver_error_type()

    def _boom(config, partition, meta):
        import adbc_driver_manager

        raise error_type("driver exploded", status_code=adbc_driver_manager.AdbcStatusCode.INTERNAL)

    monkeypatch.setattr("boti_data.db.partitioned_execution.fetch_partition_table", _boom)
    frame = _fetch(_config(ro_sqlite))
    assert len(frame) == 4
    assert any("ADBC fetch failed" in record.getMessage() for record in caplog.records)


def _local_postgres_uri() -> str | None:
    if importlib.util.find_spec("adbc_driver_postgresql") is None:
        return None
    try:
        with socket.create_connection(("localhost", 5432), timeout=1):
            pass
    except OSError:
        return None
    return f"postgresql://{getpass.getuser()}@localhost:5432/postgres"


@pytest.mark.skipif(_local_postgres_uri() is None, reason="no local PostgreSQL on 5432")
def test_postgres_partition_through_adbc_is_read_only_and_numbered(monkeypatch) -> None:
    from adbc_driver_postgresql import dbapi as pg

    uri = _local_postgres_uri()
    table = f"adbc_probe_{getpass.getuser()}"
    admin = pg.connect(uri, autocommit=True)
    with admin.cursor() as cursor:
        cursor.execute(f"drop table if exists {table}")
        cursor.execute(f"create table {table} (id bigint, name text, amount double precision)")
        cursor.execute(f"insert into {table} values (1, 'a', 1.5), (2, 'b', 2.5), (3, 'c', null)")
        cursor.execute("select count(*) from pg_stat_activity where usename = current_user")
        before = cursor.fetch_arrow_table().column(0)[0].as_py()
    try:
        # The SQLAlchemy driver suffix is only parsed, never imported, on the ADBC path.
        config = _config(uri.replace("postgresql://", "postgresql+psycopg://"))
        spec = SqlPartitionSpec(
            sql=f"select id, name, amount from {table} where id >= %(id_1)s and name like '%%' order by id",
            params={"id_1": 2},
        )
        calls: list[str] = []
        real = adbc_fetch.fetch_partition_table
        monkeypatch.setattr(
            "boti_data.db.partitioned_execution.fetch_partition_table",
            lambda c, p, m: (calls.append(p.sql), real(c, p, m))[1],
        )
        frame = SqlPartitionExecutor.fetch_partition(
            config=config,
            gate_key="pg",
            max_concurrent_fetches=1,
            partition=spec,
            meta_dtypes={"id": "Int64", "name": "string", "amount": "Float64"},
            use_arrow=True,
        )
        assert calls and frame["id"].tolist() == [2, 3]
        with (
            adbc_fetch.open_read_only(config) as connection,
            connection.cursor() as cursor,
            pytest.raises(Exception, match="(?i)read-only"),
        ):
            cursor.execute(f"insert into {table} values (4, 'd', 4.0)")
        with admin.cursor() as cursor:
            cursor.execute("select count(*) from pg_stat_activity where usename = current_user")
            after = cursor.fetch_arrow_table().column(0)[0].as_py()
        assert after == before, "an ADBC connection outlived its fetch"
    finally:
        with admin.cursor() as cursor:
            cursor.execute(f"drop table if exists {table}")
        admin.close()
