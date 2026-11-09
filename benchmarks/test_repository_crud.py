"""Case 8: 1k single-row upserts (insert then update by key) through SqlRepository."""

import itertools

import pytest
from sqlalchemy import create_engine

from boti_data import SqlDatabaseConfig, SqlRepository

ROWS = 1_000
_offsets = itertools.count(0, ROWS)


@pytest.mark.perf
def test_repository_1k_upserts(benchmark, bench_root) -> None:
    dsn = f"sqlite:///{bench_root / 'crud.db'}"
    engine = create_engine(dsn)
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "create table if not exists kv (id integer primary key, value text not null)"
        )
    engine.dispose()

    with SqlRepository(SqlDatabaseConfig(connection_url=dsn), "kv", query_only=False) as repo:

        def _upsert_block() -> int:
            start = next(_offsets) + 1
            for key in range(start, start + ROWS):
                repo.insert({"id": key, "value": "v1"})
                repo.update(key, {"value": "v2"})
            return ROWS

        assert benchmark(_upsert_block) == ROWS
