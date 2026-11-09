"""R1 Arrow-native fetch and R7 streaming: equivalence, fetcher selection, memory bound."""

from __future__ import annotations

import datetime as dt
import sqlite3
import tracemalloc

import pandas as pd
import pyarrow as pa
import pytest
import sqlalchemy as sa

from boti_data.db.arrow_fetch import (
    CursorArrowFetcher,
    NativeArrowFetcher,
    column_array,
    fetch_table,
    fetcher_for,
    iter_record_batches,
    stream_statement,
)
from boti_data.db.partitioned_execution import SqlPartitionExecutor
from boti_data.db.partitioned_types import SqlPartitionSpec
from boti_data.db.sql_config import SqlDatabaseConfig, WorkerSqlConfig

META = {
    "id": "Int64",
    "name": "string",
    "amount": "Float64",
    "flag": "boolean",
    "ts": "datetime64[ns]",
    "ts_utc": "datetime64[ns, UTC]",
    "day": "datetime64[ns]",
}


@pytest.fixture
def sqlite_db(tmp_path) -> str:
    path = tmp_path / "equiv.sqlite"
    con = sqlite3.connect(path)
    con.execute(
        "create table t (id integer, name text, amount real, flag integer, ts text, ts_utc text, day text)"
    )
    rows = [
        (1, "alpha", 1.5, 1, "2026-01-01 10:00:00", "2026-01-01T10:00:00+02:00", "2026-01-01"),
        (2, None, None, 0, "2026-01-02 11:30:00", "2026-01-02T00:00:00Z", None),
        (3, "gamma", 3.0, None, None, None, "2026-03-03"),
        (4, " ", 4.25, 1, " ", "", "2026-04-04"),
    ]
    con.executemany("insert into t values (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return str(path)


def _fetch(db: str, *, use_arrow: bool) -> pd.DataFrame:
    cfg = WorkerSqlConfig.from_database_config(
        SqlDatabaseConfig(connection_url=f"sqlite:///{db}", query_only=False)
    )
    spec = SqlPartitionSpec(
        sql="select id, name, amount, flag, ts, ts_utc, day from t order by id", params=None
    )
    return SqlPartitionExecutor.fetch_partition(
        config=cfg,
        gate_key="equiv",
        max_concurrent_fetches=1,
        partition=spec,
        meta_dtypes=META,
        use_arrow=use_arrow,
    )


def test_arrow_and_pandas_paths_produce_identical_frames(sqlite_db) -> None:
    arrow = _fetch(sqlite_db, use_arrow=True)
    pandas = _fetch(sqlite_db, use_arrow=False)
    # The pandas path parses timestamps at microsecond resolution; the canonical
    # dtype is nanoseconds, which the Arrow path produces directly.
    expected = pandas.astype(
        {"ts": "datetime64[ns]", "day": "datetime64[ns]", "ts_utc": "datetime64[ns, UTC]"}
    )
    pd.testing.assert_frame_equal(arrow, expected)
    assert arrow["ts_utc"].tolist()[:2] == [
        pd.Timestamp("2026-01-01T08:00:00Z"),
        pd.Timestamp("2026-01-02T00:00:00Z"),
    ]
    assert arrow["ts"].isna().tolist() == [False, False, True, True]
    assert arrow["ts"].tolist()[1] == pd.Timestamp("2026-01-02 11:30:00")
    assert arrow["flag"].tolist()[:2] == [True, False]


def test_sqlite_uses_cursor_fetcher_and_duckdb_uses_native(tmp_path) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'f.sqlite'}")
    with engine.connect() as conn:
        assert isinstance(fetcher_for(conn.exec_driver_sql("select 1 as a")), CursorArrowFetcher)
    pytest.importorskip("duckdb_engine")
    duck = sa.create_engine(f"duckdb:///{tmp_path / 'f.duckdb'}")
    with duck.connect() as conn:
        conn.exec_driver_sql("create table t(a bigint, ts timestamp)")
        conn.exec_driver_sql("insert into t values (1, '2026-01-01 10:00:00'), (2, NULL)")
        result = conn.exec_driver_sql("select * from t order by a")
        assert isinstance(fetcher_for(result), NativeArrowFetcher)
        table = fetch_table(result, {"a": "Int64", "ts": "datetime64[ns, UTC]"})
    assert table.schema.field("ts").type == pa.timestamp("ns", tz="UTC")
    assert table.column("a").to_pylist() == [1, 2]


def test_iter_record_batches_streams_in_batch_sized_pieces(sqlite_db) -> None:
    engine = sa.create_engine(f"sqlite:///{sqlite_db}")
    with engine.connect() as conn:
        result = conn.exec_driver_sql("select id, name from t order by id")
        sizes = [
            b.num_rows
            for b in iter_record_batches(result, {"id": "Int64", "name": "string"}, batch_size=3)
        ]
    assert sizes == [3, 1]


def test_column_array_falls_back_per_value_only_when_vectorised_cast_fails() -> None:
    field = pa.field("x", pa.int64())
    assert column_array((1.0, 2.9, None), field).to_pylist() == [1, 2, None]
    weird = column_array((dt.date(2026, 1, 1), "2026-02-02"), pa.field("d", pa.date32()))
    assert weird.to_pylist() == [dt.date(2026, 1, 1), dt.date(2026, 2, 2)]
    mixed = column_array((object(), 1), pa.field("s", pa.string()))
    assert mixed.type == pa.string() and mixed[1].as_py() == "1"


def test_stream_statement_bounds_memory_to_about_one_batch(tmp_path) -> None:
    path = tmp_path / "big.sqlite"
    con = sqlite3.connect(path)
    con.execute("create table big (id integer, payload text)")
    con.executemany("insert into big values (?, ?)", ((i, "x" * 200) for i in range(200_000)))
    con.commit()
    con.close()
    engine = sa.create_engine(f"sqlite:///{path}")
    metadata = sa.MetaData()
    big = sa.Table("big", metadata, autoload_with=engine)
    tracemalloc.start()
    rows = 0
    for batch in stream_statement(engine, sa.select(big), batch_size=10_000):
        rows += batch.num_rows
        one_batch = batch.nbytes
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert rows == 200_000
    assert peak < 6 * one_batch, f"peak {peak} vs batch {one_batch}"
