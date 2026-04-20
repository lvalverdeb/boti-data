"""SinkPipeline example: materialize a DataHelper load into a partitioned CSV dataset."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import Date, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import CsvSink, DataHelper, SinkPipeline


class Base(DeclarativeBase):
    pass


class SourceEvent(Base):
    __tablename__ = "source_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(16))


def _seed(engine) -> None:
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                SourceEvent(id=1, event_date=dt.date(2026, 4, 15), status="active"),
                SourceEvent(id=2, event_date=dt.date(2026, 4, 16), status="inactive"),
                SourceEvent(id=3, event_date=dt.date(2026, 4, 17), status="active"),
            ]
        )
        session.commit()


def main() -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        db_path = root / "csv_pipeline_example.db"
        worker_dsn_env_var = "BOTI_EXAMPLE_CSV_SQLITE_DSN"
        sqlite_dsn = f"sqlite:///{db_path}"
        previous_worker_dsn = os.environ.get(worker_dsn_env_var)
        os.environ[worker_dsn_env_var] = sqlite_dsn
        engine = create_engine(f"sqlite:///{db_path}")
        try:
            _seed(engine)
        finally:
            engine.dispose()

        helper = DataHelper(
            backend="sqlalchemy",
            connection_url=sqlite_dsn,
            worker_connection_env_var=worker_dsn_env_var,
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
            table="source_events",
        )
        sink = CsvSink(
            {
                "storage_path": str(root / "source_events_csv"),
                "partition_on": ["partition_date"],
                "project_root": root,
            }
        )
        pipeline = SinkPipeline(helper, sink, date_field="event_date")
        try:
            result = pipeline.write(filters={"status__exact": "active"})
        finally:
            pipeline.close()
            if previous_worker_dsn is None:
                os.environ.pop(worker_dsn_env_var, None)
            else:
                os.environ[worker_dsn_env_var] = previous_worker_dsn

    summary = {
        "path": result.path,
        "n_files": len(result.files),
        "sample_file": result.files[0] if result.files else None,
    }
    print(f"SinkPipeline wrote CSV dataset to {summary['path']}")
    print(f"csv files={summary['n_files']} sample={summary['sample_file']}")
    return summary


if __name__ == "__main__":
    main()
