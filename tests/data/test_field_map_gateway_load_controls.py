"""
DataGateway load controls: _has_any_rows, persist=True, timeout=, and the
load_period()/aload_period() date-range helper.

Split out of test_field_map_gateway.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest
from pydantic import SecretStr
from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.gateway import DataGateway

from .conftest import _legacy_gw

# ---------------------------------------------------------------------------
# Item 4: _has_any_rows
# ---------------------------------------------------------------------------


def test_has_any_rows_true_pandas() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    assert DataGateway._has_any_rows(df) is True


def test_has_any_rows_false_pandas() -> None:
    df = pd.DataFrame({"a": []})
    assert DataGateway._has_any_rows(df) is False


def test_has_any_rows_no_columns() -> None:
    df = pd.DataFrame()
    assert DataGateway._has_any_rows(df) is False


# ---------------------------------------------------------------------------
# Item 2: persist=True
# ---------------------------------------------------------------------------


def test_load_persist_kwarg_accepted_and_returns_df(legacy_dsn) -> None:
    """persist=True is consumed as a control kwarg; result is a valid DataFrame."""
    gw = _legacy_gw(legacy_dsn)
    try:
        df = gw.load(persist=True, as_pandas=True)
        assert len(df) == 3
    finally:
        gw.close()


# ---------------------------------------------------------------------------
# Item 3: timeout=
# ---------------------------------------------------------------------------


def test_aload_timeout_not_exceeded(legacy_dsn) -> None:
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    try:
        df = asyncio.run(gw.aload(timeout=30, as_pandas=True))
        assert len(df) == 3
    finally:
        gw.close()


def test_aload_timeout_raises_when_exceeded(legacy_dsn, monkeypatch) -> None:
    """When timeout is hit asyncio.TimeoutError propagates."""
    import asyncio

    from boti_data.gateway.configured_load import ConfiguredLoadService

    original = ConfiguredLoadService.aload

    async def _slow(self, *args, **kwargs):
        await asyncio.sleep(10)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ConfiguredLoadService, "aload", _slow)
    gw = _legacy_gw(legacy_dsn)
    try:
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(gw.aload(timeout=0.01, as_pandas=True))
    finally:
        gw.close()


# ---------------------------------------------------------------------------
# Item 1: load_period / aload_period
# ---------------------------------------------------------------------------

# We test period filtering using SemanticProduct because the "date" column
# is not present in the existing test fixtures.  We create a lightweight new
# table with a date column inline.


class DateBase(DeclarativeBase):
    pass


class EventRow(DateBase):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(32))
    event_date: Mapped[_dt.date] = mapped_column()


@pytest.fixture(scope="module")
def events_dsn(tmp_path_factory) -> str:
    db_path = tmp_path_factory.mktemp("db") / "events.db"
    engine = create_engine(f"sqlite:///{db_path}")
    DateBase.metadata.create_all(engine)
    with Session(engine) as s:
        s.add_all(
            [
                EventRow(name="alpha", event_date=_dt.date(2024, 1, 10)),
                EventRow(name="beta", event_date=_dt.date(2024, 2, 15)),
                EventRow(name="gamma", event_date=_dt.date(2024, 3, 20)),
            ]
        )
        s.commit()
    engine.dispose()
    return f"sqlite:///{db_path}"


def _event_gw(dsn: str) -> DataGateway:
    config = SqlDatabaseConfig(connection_url=SecretStr(dsn), query_only=False)
    return DataGateway(config, table="events")


def test_load_period_single_date(events_dsn) -> None:
    """Single-date period (start == end) adds a ``field__date`` filter."""
    gw = _event_gw(events_dsn)
    try:
        df = gw.load_period("event_date", "2024-02-15", "2024-02-15", as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["name"] == "beta"
    finally:
        gw.close()


def test_load_period_range(events_dsn) -> None:
    """Date-range period adds a ``field__date__range`` filter."""
    gw = _event_gw(events_dsn)
    try:
        df = gw.load_period("event_date", "2024-01-01", "2024-02-28", as_pandas=True)
        assert len(df) == 2
        names = set(df["name"].tolist())
        assert names == {"alpha", "beta"}
    finally:
        gw.close()


def test_load_period_invalid_range_raises(events_dsn) -> None:
    gw = _event_gw(events_dsn)
    try:
        with pytest.raises(ValueError, match="'start' date cannot be later"):
            gw.load_period("event_date", "2024-12-31", "2024-01-01")
    finally:
        gw.close()


def test_aload_period_range(events_dsn) -> None:
    import asyncio

    gw = _event_gw(events_dsn)
    try:
        df = asyncio.run(gw.aload_period("event_date", "2024-01-01", "2024-03-31", as_pandas=True))
        assert len(df) == 3
    finally:
        gw.close()
