"""R13: polars results are built from Arrow with no pandas detour; Parquet scans lazily."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from boti_data import DataGateway
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.parquet import ParquetDataConfig, ParquetDataResource

pl = pytest.importorskip("polars")


@pytest.fixture
def no_pandas_frames(monkeypatch):
    def _boom(self, *args, **kwargs):  # noqa: ANN001
        raise AssertionError("a pandas DataFrame was constructed on the polars path")

    monkeypatch.setattr(pd.DataFrame, "__init__", _boom)


@pytest.fixture
def orders_db(tmp_path) -> str:
    path = tmp_path / "orders.sqlite"
    con = sqlite3.connect(path)
    con.execute("create table orders (id integer primary key, customer text, total real, ts text)")
    con.executemany(
        "insert into orders values (?, ?, ?, ?)",
        ((i, f"c{i % 3}", float(i), "2026-01-01 10:00:00") for i in range(1, 51)),
    )
    con.commit()
    con.close()
    return str(path)


def test_polars_sql_load_builds_no_pandas_objects(orders_db, no_pandas_frames) -> None:
    gateway = DataGateway(
        SqlDatabaseConfig(connection_url=f"sqlite:///{orders_db}", query_only=False), table="orders"
    )
    with gateway:
        frame = gateway.load(return_type="polars", customer="c1")
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 17
    # SQLite reflects INTEGER as SQLAlchemy Integer, which the schema mapper types as int32.
    assert frame.schema["id"] == pl.Int32


def test_polars_parquet_load_builds_no_pandas_objects(tmp_path, no_pandas_frames) -> None:
    root = tmp_path / "pq"
    root.mkdir()
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table({"id": list(range(10)), "v": [float(i) for i in range(10)]}), root / "a.parquet"
    )
    gateway = DataGateway.from_backend("parquet", parquet_storage_path=str(root))
    with gateway:
        frame = gateway.load(return_type="polars")
    assert isinstance(frame, pl.DataFrame) and frame.height == 10


def test_scan_polars_is_lazy_and_reads_through_the_resource_filesystem(tmp_path) -> None:
    root = tmp_path / "pq"
    root.mkdir()
    pd.DataFrame({"id": range(25), "v": [float(i) for i in range(25)]}).to_parquet(
        root / "a.parquet", index=False
    )
    pd.DataFrame({"id": range(25, 40), "v": [float(i) for i in range(25, 40)]}).to_parquet(
        root / "b.parquet", index=False
    )
    with ParquetDataResource(ParquetDataConfig(parquet_storage_path=str(root))) as resource:
        lazy = resource.scan_polars(columns=["id"])
        assert isinstance(lazy, pl.LazyFrame)
        out = lazy.filter(pl.col("id") >= 30).collect()
    assert out.columns == ["id"] and out.height == 10
