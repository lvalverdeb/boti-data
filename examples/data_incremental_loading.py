"""Incremental loading example: DataHelper.load_incremental and ParquetPipeline.incremental."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from sqlalchemy import Date, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import (
    DataHelper,
    FileWatermarkStore,
    ParquetPipeline,
)


class Base(DeclarativeBase):
    pass


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(16))


def _seed(engine, extra_rows: bool = False) -> None:
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        if extra_rows:
            rows = [
                Event(id=4, event_date=dt.date(2026, 5, 4), status="active"),
                Event(id=5, event_date=dt.date(2026, 5, 5), status="inactive"),
            ]
        else:
            rows = [
                Event(id=1, event_date=dt.date(2026, 5, 1), status="active"),
                Event(id=2, event_date=dt.date(2026, 5, 2), status="inactive"),
                Event(id=3, event_date=dt.date(2026, 5, 3), status="active"),
            ]
        session.add_all(rows)
        session.commit()


def _open_events_helper(sqlite_dsn: str) -> DataHelper:
    return DataHelper(
        backend="sqlalchemy",
        connection_url=sqlite_dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="events",
    )


def _run_incremental_pass(helper: DataHelper, watermark_store: FileWatermarkStore) -> Any:
    return helper.pandas.load_incremental(
        watermark_field="event_date",
        watermark_source="events_table",
        watermark_store=watermark_store,
    )


def _run_first_and_second_passes(
    sqlite_dsn: str, watermark_store: FileWatermarkStore
) -> dict[str, object]:
    """DataHelper.load_incremental — first run (full load), then a no-op rerun."""
    with _open_events_helper(sqlite_dsn) as helper:
        result = _run_incremental_pass(helper, watermark_store)
        # Second run with same data — should return empty (no new rows)
        result2 = _run_incremental_pass(helper, watermark_store)

    return {
        "first_run_count": result.records_loaded,
        "first_run_watermark": str(result.current_watermark),
        "first_run_committed": result.watermark_committed,
        "second_run_count": result2.records_loaded,
    }


def _run_third_pass_after_new_data(
    sqlite_dsn: str, watermark_store: FileWatermarkStore
) -> dict[str, object]:
    """Add new data and run again — should pick up only new rows."""
    engine = create_engine(sqlite_dsn)
    try:
        _seed(engine, extra_rows=True)
    finally:
        engine.dispose()

    with _open_events_helper(sqlite_dsn) as helper:
        result3 = _run_incremental_pass(helper, watermark_store)

    return {
        "third_run_count": result3.records_loaded,
        "third_run_watermark": str(result3.current_watermark),
        "third_run_committed": result3.watermark_committed,
    }


def _run_pipeline_incremental(sqlite_dsn: str, root: Path) -> dict[str, object]:
    """ParquetPipeline.incremental — end-to-end incremental materialize."""
    watermark_store2 = FileWatermarkStore(str(root / "watermarks2.json"))
    with _open_events_helper(sqlite_dsn) as helper:
        pipeline = ParquetPipeline.incremental(
            helper,
            {
                "backend": "parquet",
                "storage_path": str(root / "events_dataset"),
                "project_root": root,
            },
            watermark_field="event_date",
            watermark_store=watermark_store2,
            watermark_source="events_pipeline",
            date_field="event_date",
        )

        # First materialize — loads all 5 rows
        r1 = pipeline.materialize(reload=True)
        pipeline_first_rows = int(r1.frame.compute().shape[0]) if r1.frame is not None else 0

        # Second materialize without new data — should be empty
        r2 = pipeline.materialize(reload=True)
        pipeline_second_rows = int(r2.frame.compute().shape[0]) if r2.frame is not None else 0

        pipeline.parquet_sink.close()

    return {
        "pipeline_first_rows": pipeline_first_rows,
        "pipeline_second_rows": pipeline_second_rows,
    }


def main() -> dict[str, object]:
    results: dict[str, object] = {}
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        db_path = root / "incremental.db"
        watermark_path = root / "watermarks.json"
        sqlite_dsn = f"sqlite:///{db_path}"

        engine = create_engine(sqlite_dsn)
        try:
            _seed(engine, extra_rows=False)
        finally:
            engine.dispose()

        watermark_store = FileWatermarkStore(str(watermark_path))

        results.update(_run_first_and_second_passes(sqlite_dsn, watermark_store))
        results.update(_run_third_pass_after_new_data(sqlite_dsn, watermark_store))
        results.update(_run_pipeline_incremental(sqlite_dsn, root))

    return results


if __name__ == "__main__":
    results = main()
    for key, value in results.items():
        print(f"  {key}: {value}")
