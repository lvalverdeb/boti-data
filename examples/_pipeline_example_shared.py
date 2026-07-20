"""Shared SQLAlchemy model and DB-seeding helper for the sink-pipeline examples.

Split out purely to deduplicate an identical setup copied across
data_csv_sink_pipeline.py, data_jsonl_sink_pipeline.py, and
data_parquet_pipeline.py. Leading underscore keeps it out of any
"public example" listing.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Date, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


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


def _seed_source_events_db(sqlite_dsn: str) -> None:
    engine = create_engine(sqlite_dsn)
    try:
        _seed(engine)
    finally:
        engine.dispose()
