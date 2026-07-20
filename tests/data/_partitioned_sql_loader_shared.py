"""Shared SQLAlchemy models and DB-seeding helpers for the partitioned SQL loader tests.

Split out of test_partitioned_sql_loader.py purely for god-module/long-file
headroom. Leading underscore so pytest does not collect this as a test module.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Date, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data.db import SqlDatabaseConfig


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "partitioned_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(String(128))


class Event(Base):
    __tablename__ = "partitioned_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date)


def _create_user_db(tmp_path, rows: list[dict[str, object]]) -> SqlDatabaseConfig:
    db_path = tmp_path / "partitioned_loader.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(**row) for row in rows])
            session.commit()
    finally:
        engine.dispose()

    return SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )


def _create_event_db(tmp_path, rows: list[dict[str, object]]) -> SqlDatabaseConfig:
    db_path = tmp_path / "partitioned_events.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([Event(**row) for row in rows])
            session.commit()
    finally:
        engine.dispose()

    return SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
