"""
DataHelper load/aload/period/join/session tests: async parquet loading,
sync-context aload bridges, running-event-loop guards, period loading,
semi-join, left-join helpers, and the managed dask session context manager.

Split out of test_helper.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import datetime as dt

import dask.dataframe as dd
import pandas as pd
import pytest
from sqlalchemy import Date, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataHelper
from boti_data.parquet import ParquetDataConfig


@pytest.mark.asyncio
async def test_helper_aloads_parquet(temp_project_root) -> None:
    file_path = temp_project_root / "data" / "helper_users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(
        file_path, index=False
    )

    helper = DataHelper(
        ParquetDataConfig(
            project_root=temp_project_root,
            parquet_storage_path=str(file_path.parent),
            parquet_filename="helper_users",
        )
    )
    try:
        frame = await helper.aload(filters={"status__exact": "active"})
    finally:
        await helper.aclose()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1]


def test_helper_aload_sync_runs_from_sync_context(temp_project_root) -> None:
    file_path = temp_project_root / "data" / "helper_users_sync_bridge.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(
        file_path, index=False
    )

    helper = DataHelper(
        ParquetDataConfig(
            project_root=temp_project_root,
            parquet_storage_path=str(file_path.parent),
            parquet_filename="helper_users_sync_bridge",
        )
    )
    try:
        frame = helper.aload_sync(filters={"status__exact": "active"})
    finally:
        helper.close()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1]


@pytest.mark.asyncio
async def test_helper_aload_sync_rejects_running_event_loop(temp_project_root) -> None:
    file_path = temp_project_root / "data" / "helper_users_loop_guard.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1], "status": ["active"]}).to_parquet(file_path, index=False)

    helper = DataHelper(
        ParquetDataConfig(
            project_root=temp_project_root,
            parquet_storage_path=str(file_path.parent),
            parquet_filename="helper_users_loop_guard",
        )
    )
    try:
        with pytest.raises(RuntimeError, match=r"Use `await helper.aload\(\.\.\.\)`"):
            helper.aload_sync(filters={"status__exact": "active"})
    finally:
        await helper.aclose()


@pytest.mark.asyncio
async def test_engine_bound_aload_sync_rejects_running_event_loop(temp_project_root) -> None:
    file_path = temp_project_root / "data" / "helper_engine_loop_guard.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1], "status": ["active"]}).to_parquet(file_path, index=False)

    helper = DataHelper(
        ParquetDataConfig(
            project_root=temp_project_root,
            parquet_storage_path=str(file_path.parent),
            parquet_filename="helper_engine_loop_guard",
        )
    )
    try:
        with pytest.raises(RuntimeError, match=r"Use `await helper.aload\(\.\.\.\)`"):
            helper.pandas.aload_sync(filters={"status__exact": "active"})
    finally:
        await helper.aclose()


def test_helper_load_period_uses_gateway_period_loader(tmp_path) -> None:
    class Base(DeclarativeBase):
        pass

    class Event(Base):
        __tablename__ = "helper_events"

        id: Mapped[int] = mapped_column(primary_key=True)
        event_date: Mapped[dt.date] = mapped_column(Date())
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "helper_period.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    Event(event_date=dt.date(2026, 1, 1), status="active"),
                    Event(event_date=dt.date(2026, 1, 2), status="inactive"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    helper = DataHelper(
        {
            "backend": "sqlalchemy",
            "connection_url": f"sqlite:///{db_path}",
            "poolclass": "sqlalchemy.pool.NullPool",
            "query_only": False,
            "table": "helper_events",
            "df_params": {"fieldnames": ("event_date", "status")},
        }
    )
    try:
        frame = helper.load_period("event_date", "2026-01-01", "2026-01-01", as_pandas=True)
    finally:
        helper.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active"]


def test_helper_aload_period_sync_uses_gateway_period_loader(tmp_path) -> None:
    class Base(DeclarativeBase):
        pass

    class Event(Base):
        __tablename__ = "helper_events_async_bridge"

        id: Mapped[int] = mapped_column(primary_key=True)
        event_date: Mapped[dt.date] = mapped_column(Date())
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "helper_period_sync_bridge.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    Event(event_date=dt.date(2026, 1, 1), status="active"),
                    Event(event_date=dt.date(2026, 1, 2), status="inactive"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    helper = DataHelper(
        {
            "backend": "sqlalchemy",
            "connection_url": f"sqlite:///{db_path}",
            "poolclass": "sqlalchemy.pool.NullPool",
            "query_only": False,
            "table": "helper_events_async_bridge",
        }
    )
    try:
        frame = helper.aload_period_sync("event_date", "2026-01-01", "2026-01-01", as_pandas=True)
    finally:
        helper.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active"]


def test_helper_asemi_join_sync_runs_from_sync_context(tmp_path) -> None:
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "helper_users_async_bridge"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "helper_asemijoin_sync_bridge.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(id=1, status="active"), User(id=2, status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="helper_users_async_bridge",
    )
    try:
        frame = helper.asemi_join_sync(pd.Series([1]), on="id")
    finally:
        helper.close()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute().sort_values("id")["id"].tolist() == [1]


def test_helper_left_join_uses_indexed_join_for_join_key() -> None:
    left = dd.from_pandas(
        pd.DataFrame({"id": pd.Series([1, 2, 3], dtype="Int64"), "left_value": ["a", "b", "c"]}),
        npartitions=2,
    )
    right = dd.from_pandas(
        pd.DataFrame({"id": pd.Series([1, 3], dtype="Int64"), "right_value": ["x", "z"]}),
        npartitions=1,
    )

    joined = DataHelper.left_join(
        left,
        right,
        join_key="id",
        join_schema_map={"id": "Int64"},
        persist=True,
    )
    computed = joined.compute().sort_values("id").reset_index(drop=True)

    assert computed.loc[0, "right_value"] == "x"
    assert pd.isna(computed.loc[1, "right_value"])
    assert computed.loc[2, "right_value"] == "z"


def test_helper_left_join_uses_column_join_when_join_key_missing() -> None:
    left = pd.DataFrame({"id": pd.Series([1, 2], dtype="Int64"), "left_value": ["a", "b"]})
    right = pd.DataFrame({"id": pd.Series([1], dtype="Int64"), "right_value": ["x"]})

    joined = DataHelper.left_join(
        left,
        right,
        left_on=["id"],
        join_schema_map={"id": "Int64"},
    )

    assert joined["right_value"].tolist()[0] == "x"
    assert pd.isna(joined["right_value"].tolist()[1])


def test_helper_session_creates_managed_dask_session() -> None:
    pytest.importorskip("dask.distributed")

    with DataHelper.session(
        cluster_kwargs={
            "n_workers": 1,
            "threads_per_worker": 1,
            "processes": False,
            "dashboard_address": ":0",
        }
    ) as client:
        summary = client.scheduler_info()

    assert summary["workers"]
