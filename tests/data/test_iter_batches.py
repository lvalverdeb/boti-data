"""R7: iter_batches on the repository and the gateway never materialise the result."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pyarrow as pa
import pytest

from boti_data import DataGateway, SqlRepository
from boti_data.db.sql_config import SqlDatabaseConfig


@pytest.fixture
def orders_db(tmp_path) -> str:
    path = tmp_path / "orders.sqlite"
    con = sqlite3.connect(path)
    con.execute("create table orders (id integer primary key, customer text, total real)")
    con.executemany(
        "insert into orders values (?, ?, ?)", ((i, f"c{i % 7}", float(i)) for i in range(1, 251))
    )
    con.commit()
    con.close()
    return str(path)


def test_repository_iter_batches_streams_typed_batches(orders_db) -> None:
    with SqlRepository(
        SqlDatabaseConfig(connection_url=f"sqlite:///{orders_db}"), "orders", query_only=False
    ) as repo:
        batches = list(repo.iter_batches(batch_size=100))
    assert [b.num_rows for b in batches] == [100, 100, 50]
    assert batches[0].schema.field("id").type == pa.int64()
    assert pa.Table.from_batches(batches).column("id").to_pylist()[-1] == 250


@pytest.mark.asyncio
async def test_async_repository_iter_batches(orders_db) -> None:
    pytest.importorskip("aiosqlite")
    from boti_data import AsyncSqlRepository

    async with AsyncSqlRepository(
        SqlDatabaseConfig(connection_url=f"sqlite+aiosqlite:///{orders_db}"),
        "orders",
        query_only=False,
    ) as repo:
        sizes = [b.num_rows async for b in repo.iter_batches(columns=["id"], batch_size=120)]
    assert sizes == [120, 120, 10]


def test_gateway_iter_batches_configured_mode_applies_filters_and_field_map(orders_db) -> None:
    gateway = DataGateway(
        SqlDatabaseConfig(connection_url=f"sqlite:///{orders_db}", query_only=False),
        table="orders",
        field_map={"customer": "customer_code"},
    )
    with gateway:
        batches = list(gateway.iter_batches(customer_code="c1", batch_size=20))
    table = pa.Table.from_batches(batches)
    assert "customer_code" in table.column_names
    assert set(table.column("customer_code").to_pylist()) == {"c1"}
    assert table.num_rows == 36


def test_gateway_iter_batches_structured_sql_text(orders_db) -> None:
    gateway = DataGateway(
        SqlDatabaseConfig(connection_url=f"sqlite:///{orders_db}", query_only=False)
    )
    with gateway:
        frames = [
            b.to_pandas()
            for b in gateway.iter_batches(
                sql="select id, total from orders where id <= 5", batch_size=2
            )
        ]
    out = pd.concat(frames, ignore_index=True)
    assert out["id"].tolist() == [1, 2, 3, 4, 5]


def test_gateway_iter_batches_parquet(tmp_path) -> None:
    root = tmp_path / "pq"
    root.mkdir()
    pd.DataFrame({"id": range(10), "v": [float(i) for i in range(10)]}).to_parquet(
        root / "a.parquet", index=False
    )
    pd.DataFrame({"id": range(10, 20), "v": [float(i) for i in range(10, 20)]}).to_parquet(
        root / "b.parquet", index=False
    )
    gateway = DataGateway.from_backend("parquet", parquet_storage_path=str(root))
    with gateway:
        total = sum(b.num_rows for b in gateway.iter_batches(columns=["id"], batch_size=4))
    assert total == 20
