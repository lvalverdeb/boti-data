"""Deterministic fixtures for the benchmark suite: a 1M-row SQLite table, a
matching hive-partitioned Parquet dataset and in-memory frames. Everything is
local so the suite runs on a CI runner with no services."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROWS_1M = 1_000_000
ROWS_100K = 100_000
GROUPS = 100
INSERT_BATCH = 50_000


def event_frame(start: int, stop: int) -> pd.DataFrame:
    ids = pd.RangeIndex(start, stop)
    return pd.DataFrame(
        {
            "id": ids,
            "event_date": [f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in ids],
            "group_id": (ids % GROUPS).astype("int64"),
            "amount": (ids % 997).astype("float64") / 10.0,
            "payload": [f"payload-{i}" for i in ids],
        }
    )


def seed_sqlite(db_path: Path, rows: int = ROWS_1M) -> str:
    """Create ``events`` with *rows* rows; return the DSN."""
    dsn = f"sqlite:///{db_path}"
    engine = create_engine(dsn)
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "create table events (id integer primary key, event_date text not null, "
                "group_id integer not null, amount real not null, payload text not null)"
            )
            conn.exec_driver_sql("create index idx_events_group on events(group_id)")
            insert = text(
                "insert into events(id, event_date, group_id, amount, payload) "
                "values (:id, :event_date, :group_id, :amount, :payload)"
            )
            for start in range(1, rows + 1, INSERT_BATCH):
                stop = min(start + INSERT_BATCH, rows + 1)
                conn.execute(insert, event_frame(start, stop).to_dict("records"))
    finally:
        engine.dispose()
    return dsn


def seed_parquet(root: Path, rows: int = ROWS_1M, files: int = 12) -> Path:
    """Write *rows* rows as a hive-partitioned ``partition_date=...`` dataset."""
    per_file = rows // files
    for index in range(files):
        partition = root / f"partition_date=2026-{index + 1:02d}-01"
        partition.mkdir(parents=True, exist_ok=True)
        start = index * per_file + 1
        stop = rows + 1 if index == files - 1 else start + per_file
        event_frame(start, stop).to_parquet(partition / "part.parquet", index=False)
    return root
