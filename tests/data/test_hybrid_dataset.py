from __future__ import annotations

import datetime as dt

import dask.dataframe as dd
import pandas as pd
import pytest
from sqlalchemy import Date, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataHelper, HybridDataset


class Base(DeclarativeBase):
    pass


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


def _build_hybrid(tmp_path) -> HybridDataset:
    db_path = tmp_path / "hybrid_dataset.db"
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


def test_hybrid_dataset_load_routes_mixed_ranges_to_dask(tmp_path) -> None:
    dataset = _build_hybrid(tmp_path)
    try:
        frame = dataset.load(start="2026-04-15", end="2026-04-20")
        computed = frame.compute().sort_values("id").reset_index(drop=True)
    finally:
        dataset.close()

    assert isinstance(frame, dd.DataFrame)
    assert computed["id"].tolist() == [1, 2, 10, 11]
    assert computed["status"].tolist() == ["hist", "hist", "live", "live"]


def test_hybrid_dataset_load_routes_single_source_when_window_is_live_only(tmp_path) -> None:
    dataset = _build_hybrid(tmp_path)
    try:
        frame = dataset.load(start="2026-04-18", end="2026-04-20", return_type="auto")
    finally:
        dataset.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame.sort_values("id")["id"].tolist() == [10, 11]


@pytest.mark.asyncio
async def test_hybrid_dataset_aload_supports_explicit_eager_return_types(tmp_path) -> None:
    dataset = _build_hybrid(tmp_path)
    try:
        frame = await dataset.aload(
            start="2026-04-15",
            end="2026-04-20",
            return_type="pandas",
        )
    finally:
        await dataset.aclose()

    assert isinstance(frame, pd.DataFrame)
    assert frame.sort_values("id")["id"].tolist() == [1, 2, 10, 11]
