"""R5: DataGateway.load(since=...) and the SQL watermark store."""

from __future__ import annotations

import sqlite3

import pytest
import sqlalchemy as sa

from boti_data import DataGateway
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.watermark import FileWatermarkStore, SqlWatermarkStore


@pytest.fixture
def events_db(tmp_path) -> str:
    path = tmp_path / "events.sqlite"
    con = sqlite3.connect(path)
    con.execute("create table events (id integer primary key, seq integer, name text)")
    con.executemany("insert into events values (?, ?, ?)", ((i, i, f"e{i}") for i in range(1, 11)))
    con.commit()
    con.close()
    return str(path)


def _gateway(db: str) -> DataGateway:
    return DataGateway(
        SqlDatabaseConfig(connection_url=f"sqlite:///{db}", query_only=False), table="events"
    )


def test_load_since_filters_by_watermark_and_advances_it(events_db, tmp_path) -> None:
    store = FileWatermarkStore(str(tmp_path / "wm.json"))
    with _gateway(events_db) as gateway:
        first = gateway.load(
            since="seq", watermark_store=store, as_pandas=True, return_type="pandas"
        )
        assert len(first) == 10
        assert store.read(source="events") == 10
        second = gateway.load(
            since="seq", watermark_store=store, as_pandas=True, return_type="pandas"
        )
        assert len(second) == 0
    con = sqlite3.connect(events_db)
    con.execute("insert into events values (11, 11, 'e11')")
    con.commit()
    con.close()
    with _gateway(events_db) as gateway:
        third = gateway.load(since=("seq", store, "events"), as_pandas=True, return_type="pandas")
    assert third["seq"].tolist() == [11]
    assert store.read(source="events") == 11


@pytest.mark.asyncio
async def test_aload_since_uses_the_same_path(events_db, tmp_path) -> None:
    store = FileWatermarkStore(str(tmp_path / "wm.json"))
    store.write(source="events", value=7)
    async with _gateway(events_db) as gateway:
        frame = await gateway.aload(
            since="seq", watermark_store=store, as_pandas=True, return_type="pandas"
        )
    assert sorted(frame["seq"].tolist()) == [8, 9, 10]
    assert store.read(source="events") == 10


def test_since_requires_a_store_and_a_source(events_db) -> None:
    with _gateway(events_db) as gateway:
        with pytest.raises(ValueError, match="watermark_store"):
            gateway.load(since="seq", as_pandas=True, return_type="pandas")
    with DataGateway(
        SqlDatabaseConfig(connection_url=f"sqlite:///{events_db}", query_only=False)
    ) as gateway:
        with pytest.raises(ValueError, match="watermark_source"):
            gateway.load(
                since="seq", watermark_store=FileWatermarkStore("/dev/null/x"), sql="select 1"
            )


def test_sql_watermark_store_round_trips_next_to_the_data(events_db) -> None:
    engine = sa.create_engine(f"sqlite:///{events_db}")
    store = SqlWatermarkStore(engine)
    assert store.read(source="events") is None
    store.write(source="events", value={"seq": 10, "at": "2026-11-09"})
    assert store.read(source="events") == {"seq": 10, "at": "2026-11-09"}
    store.write(source="events", value=11)
    assert store.read(source="events") == 11
    with engine.connect() as conn:
        rows = conn.execute(sa.text("select count(*) from boti_watermarks")).scalar_one()
    assert rows == 1
    store.clear(source="events")
    assert store.read(source="events") is None
    engine.dispose()
