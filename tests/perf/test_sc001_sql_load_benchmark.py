"""
SC-001: Load 1M rows from SQLite into a Dask DataFrame in under 10 seconds.

Goal — no automated performance gate currently enforces this threshold.
This is the gate. Run explicitly with: pytest tests/perf/ -m perf
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from sqlalchemy import create_engine, text

from boti_data import DataHelper

ROW_COUNT = 1_000_000


def _seed_sqlite(db_path: Path) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "create table events (id integer primary key, event_date text not null, "
                "group_id integer not null, payload text not null)"
            )
            conn.exec_driver_sql("create index idx_events_date on events(event_date)")
            batch_size = 10_000
            for start in range(1, ROW_COUNT + 1, batch_size):
                stop = min(start + batch_size, ROW_COUNT + 1)
                rows = [
                    {
                        "id": i,
                        "event_date": f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                        "group_id": i % 100,
                        "payload": f"payload-{i}",
                    }
                    for i in range(start, stop)
                ]
                conn.execute(
                    text(
                        "insert into events(id, event_date, group_id, payload) "
                        "values (:id, :event_date, :group_id, :payload)"
                    ),
                    rows,
                )
    finally:
        engine.dispose()


@pytest.mark.perf
def test_sc001_sql_load_1m_rows_under_10s(benchmark) -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "sc001_benchmark.db"
        _seed_sqlite(db_path)

        helper = DataHelper(
            backend="sqlalchemy",
            connection_url=f"sqlite:///{db_path}",
            worker_connection_env_var="SC001_BENCH_DB_DSN",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
            table="events",
        )
        try:
            os.environ["SC001_BENCH_DB_DSN"] = f"sqlite:///{db_path}"

            def _load() -> int:
                df = helper.dask.load()
                return len(df)

            result = benchmark(_load)
            assert result == ROW_COUNT, f"Expected {ROW_COUNT} rows, got {result}"
        finally:
            helper.close()
