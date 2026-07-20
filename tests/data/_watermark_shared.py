"""Shared SQLAlchemy models and DB-seeding helpers for the watermark tests.

Split out of test_watermark.py purely for god-module/long-file headroom.
Leading underscore so pytest does not collect this as a test module.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import Date, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class IncEvent(Base):
    __tablename__ = "inc_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(16))


class PipelineEvent(Base):
    __tablename__ = "pipeline_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(16))


def _create_event_db(tmp_path, rows: list[dict[str, Any]]) -> str:
    db_path = tmp_path / "inc_events.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([IncEvent(**r) for r in rows])
            session.commit()
    finally:
        engine.dispose()
    return f"sqlite:///{db_path}"


def _create_pipeline_db(tmp_path, rows: list[dict[str, Any]]) -> str:
    db_path = tmp_path / "pipeline.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([PipelineEvent(**r) for r in rows])
            session.commit()
    finally:
        engine.dispose()
    return f"sqlite:///{db_path}"
