"""Shared SQLAlchemy models and DataHelper/HybridDataset builders for the pipelines tests.

Split out of test_pipelines.py purely for god-module/long-file headroom.
Leading underscore so pytest does not collect this as a test module.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Date, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataHelper, HybridDataset


class Base(DeclarativeBase):
    pass


class SourceEvent(Base):
    __tablename__ = "source_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(32))


class HistoricalEvent(Base):
    __tablename__ = "historical_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(32))


class LiveEvent(Base):
    __tablename__ = "live_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(32))


def _build_source_helper(tmp_path) -> DataHelper:
    db_path = tmp_path / "pipeline_source.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
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
    finally:
        engine.dispose()

    return DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="source_events",
    )


def _build_hybrid_dataset(tmp_path) -> HybridDataset:
    db_path = tmp_path / "pipeline_hybrid.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    HistoricalEvent(id=1, event_date=dt.date(2026, 4, 15), status="hist"),
                    HistoricalEvent(id=2, event_date=dt.date(2026, 4, 17), status="hist"),
                    LiveEvent(id=10, event_date=dt.date(2026, 4, 18), status="live"),
                    LiveEvent(id=11, event_date=dt.date(2026, 4, 20), status="live"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    historical_helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="historical_events",
    )
    live_helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="live_events",
    )
    return HybridDataset(
        historical_helper,
        live_helper,
        date_field="event_date",
        split_date="2026-04-18",
    )
