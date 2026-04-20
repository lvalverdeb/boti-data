"""ParquetPipeline example: materialize a DataHelper load and optionally reload parquet."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import Date, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataHelper, ParquetPipeline


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
        db_path = root / "pipeline_example.db"
        worker_dsn_env_var = "BOTI_EXAMPLE_SQLITE_DSN"
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
        pipeline = ParquetPipeline(
            helper,
            {
                "backend": "parquet",
                "storage_path": str(root / "source_events_dataset"),
                "project_root": root,
                "partition_on": ["partition_date"],
            },
            date_field="event_date",
        )
        try:
            write_only = pipeline.materialize(filters={"status__exact": "active"})
            reloaded = pipeline.materialize(
                filters={"status__exact": "active"},
                reload=True,
                reload_options={"filters": {"partition_date__exact": "2026-04-17"}},
            )
            reloaded_rows = int(reloaded.frame.compute().shape[0]) if reloaded.frame is not None else 0
        finally:
            pipeline.close()
            if previous_worker_dsn is None:
                os.environ.pop(worker_dsn_env_var, None)
            else:
                os.environ[worker_dsn_env_var] = previous_worker_dsn

    result = {
        "path": write_only.path,
        "write_only_reloaded": write_only.reloaded,
        "reload_reloaded": reloaded.reloaded,
        "reload_rows": reloaded_rows,
    }
    print(f"ParquetPipeline wrote dataset to {result['path']}")
    print(f"write_only.reloaded={result['write_only_reloaded']}")
    print(f"reloaded.reloaded={result['reload_reloaded']} rows={result['reload_rows']}")
    return result


if __name__ == "__main__":
    main()
