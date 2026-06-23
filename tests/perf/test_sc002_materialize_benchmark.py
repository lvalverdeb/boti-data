"""
SC-002: Materialize 100K rows from SQL to partitioned Parquet.

Goal — no automated performance gate currently enforces this threshold.
This is the gate. Run explicitly with: pytest tests/perf/ -m perf
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from sqlalchemy import create_engine, text

from boti_data import DataHelper, ParquetPipeline

ROW_COUNT = 100_000


def _seed_sqlite(db_path: Path) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "create table events (id integer primary key, event_date text not null, "
                "group_id integer not null, payload text not null)"
            )
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
def test_sc002_materialize_100k_rows(benchmark) -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "sc002_benchmark.db"
        parquet_dir = Path(tmp_dir) / "parquet_out"
        _seed_sqlite(db_path)

        source = DataHelper(
            backend="sqlalchemy",
            connection_url=f"sqlite:///{db_path}",
            worker_connection_env_var="SC002_BENCH_DB_DSN",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
            table="events",
        )
        try:
            import os
            os.environ["SC002_BENCH_DB_DSN"] = f"sqlite:///{db_path}"

            pipeline = ParquetPipeline(
                source,
                {
                    "storage_path": str(parquet_dir),
                    "project_root": str(tmp_dir),
                    "partition_on": ["event_date"],
                },
                date_field="event_date",
            )
            try:
                def _materialize() -> int:
                    result = pipeline.materialize(reload=True, overwrite=True)
                    return len(result.frame) if result.frame is not None else 0

                row_count = benchmark(_materialize)
                assert row_count == ROW_COUNT, (
                    f"Expected {ROW_COUNT} rows after materialize, got {row_count}"
                )

                written = list(parquet_dir.rglob("*.parquet"))
                assert len(written) > 0, "No parquet files were written"
            finally:
                pipeline.parquet_sink.close()
                pipeline.close()
        finally:
            source.close()
