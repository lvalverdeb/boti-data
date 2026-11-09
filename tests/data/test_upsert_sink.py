"""R5: UpsertSink updates overlapping keys instead of duplicating rows."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest
import sqlalchemy as sa

from boti_data.pipelines import UpsertSink, available_sinks, create_sink
from boti_data.pipelines.sinks_upsert import upsert_statement


def _prepare(url: str) -> None:
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            sa.text("create table items (id integer primary key, name varchar(32), qty integer)")
        )
    engine.dispose()


@pytest.mark.parametrize("backend", ["sqlite", "duckdb"])
def test_second_write_updates_overlapping_keys(tmp_path, backend) -> None:
    if backend == "duckdb":
        pytest.importorskip("duckdb_engine")
    url = f"{backend}:///{tmp_path / f'items.{backend}'}"
    _prepare(url)
    first = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "qty": [1, 1, 1]})
    second = pa.table({"id": [2, 3, 4], "name": ["B", "C", "d"], "qty": [2, 2, 2]})
    with UpsertSink({"connection_url": url}, table="items", key_columns=["id"]) as sink:
        sink.write(first)
        sink.write(second)
    engine = sa.create_engine(url)
    with engine.connect() as conn:
        rows = conn.execute(sa.text("select id, name, qty from items order by id")).all()
    engine.dispose()
    assert rows == [(1, "a", 1), (2, "B", 2), (3, "C", 2), (4, "d", 2)]


def test_dask_frame_is_written_partition_by_partition(tmp_path) -> None:
    dd = pytest.importorskip("dask.dataframe")
    url = f"sqlite:///{tmp_path / 'd.sqlite'}"
    _prepare(url)
    frame = dd.from_pandas(
        pd.DataFrame({"id": range(1, 21), "name": ["x"] * 20, "qty": range(20)}), npartitions=4
    )
    with UpsertSink(
        {"connection_url": url}, table="items", key_columns=["id"], chunk_rows=3
    ) as sink:
        sink.write(frame)
        sink.write(frame)
    engine = sa.create_engine(url)
    with engine.connect() as conn:
        assert conn.execute(sa.text("select count(*) from items")).scalar_one() == 20
    engine.dispose()


def test_registry_builds_an_upsert_sink(tmp_path) -> None:
    assert "upsert" in available_sinks()
    url = f"sqlite:///{tmp_path / 'r.sqlite'}"
    _prepare(url)
    sink = create_sink("upsert", {"connection_url": url, "table": "items", "key_columns": ["id"]})
    try:
        assert isinstance(sink, UpsertSink)
    finally:
        sink.close()


def test_statement_shapes_per_dialect() -> None:
    table = sa.Table(
        "t",
        sa.MetaData(),
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("v", sa.Integer),
    )
    pg = str(
        upsert_statement(table, "postgresql", ["id"], ["id", "v"]).compile(
            dialect=__import__("sqlalchemy.dialects.postgresql", fromlist=["dialect"]).dialect()
        )
    )
    assert "ON CONFLICT (id) DO UPDATE SET v = excluded.v" in pg
    my = str(
        upsert_statement(table, "mysql", ["id"], ["id", "v"]).compile(
            dialect=__import__("sqlalchemy.dialects.mysql", fromlist=["dialect"]).dialect()
        )
    )
    assert "ON DUPLICATE KEY UPDATE" in my
    duck = str(upsert_statement(table, "duckdb", ["id"], ["id", "v"]))
    assert duck.startswith("INSERT OR REPLACE INTO")
    with pytest.raises(ValueError, match="dialect"):
        upsert_statement(table, "oracle", ["id"], ["id"])
