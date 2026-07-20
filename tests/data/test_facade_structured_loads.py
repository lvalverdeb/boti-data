"""
Data gateway tests: basic structured (SQL/parquet) loads, high-level filters,
pandas opt-in reads, and column push-down.

Split out of test_facade.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data.db import SqlDatabaseConfig
from boti_data.gateway import DataGateway
from boti_data.parquet import ParquetDataConfig, ParquetDataResource


class StubLogger:
    def debug(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def error(self, *_args, **_kwargs) -> None:
        pass


def test_facade_loads_sql_with_statement_and_filters(tmp_path) -> None:
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))
        description: Mapped[str] = mapped_column(String(128))

    db_path = tmp_path / "facade_sql_filters.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    User(status="active", description="urgent order"),
                    User(status="inactive", description="backlog item"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        frame = facade.load(
            statement=select(User),
            model=User,
            filters={"status__exact": "active", "description__icontains": "urgent"},
        )

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["status"].tolist() == ["active"]


def test_facade_can_opt_into_pandas_sql_reads() -> None:
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "pandas_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))
        description: Mapped[str] = mapped_column(String(128))

    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        Base.metadata.create_all(facade.resource.engine)
        with Session(facade.resource.engine) as session:
            session.add(User(status="active", description="urgent order"))
            session.commit()

        frame = facade.load(
            statement=select(User),
            model=User,
            as_pandas=True,
        )

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active"]


def test_facade_loads_parquet_with_high_level_filters(temp_project_root) -> None:
    file_path = temp_project_root / "data" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"id": [1, 2], "status": ["active", "inactive"], "description": ["urgent", "routine"]}
    ).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="users",
    )

    with DataGateway(config) as facade:
        frame = facade.load(filters={"status__exact": "active"})

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1]


def test_facade_can_opt_into_pandas_parquet_reads(temp_project_root) -> None:
    file_path = temp_project_root / "data" / "events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3]}).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="events",
    )

    with DataGateway(config) as facade:
        frame = facade.load(as_pandas=True, limit=2)

    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [1, 2]


def test_facade_parquet_columns_push_down_to_reader(temp_project_root, monkeypatch) -> None:
    file_path = temp_project_root / "data" / "projected_events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "status": ["active", "inactive", "active"],
            "description": ["a", "b", "c"],
        }
    ).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="projected_events",
    )

    seen_columns: list[list[str] | None] = []
    real_load_files = ParquetDataResource.load_files

    def tracking_load_files(self, filters=None, *, columns=None) -> dd.DataFrame:
        seen_columns.append(columns)
        return real_load_files(self, filters, columns=columns)

    monkeypatch.setattr(ParquetDataResource, "load_files", tracking_load_files)

    with DataGateway(config) as facade:
        frame = facade.load(filters={"status__exact": "active"}, columns=["id"])

    assert isinstance(frame, dd.DataFrame)
    assert list(frame.columns) == ["id"]
    assert frame.compute()["id"].tolist() == [1, 3]
    assert seen_columns == [["id"]]


def test_facade_pandas_parquet_reads_use_arrow_eager_path(temp_project_root, monkeypatch) -> None:
    file_path = temp_project_root / "data" / "events_arrow_pandas.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "status": ["active", "inactive", "active"]}).to_parquet(
        file_path, index=False
    )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="events_arrow_pandas",
    )

    arrow_calls: list[bool] = []
    dask_calls: list[bool] = []
    real_load_filtered_arrow = ParquetDataResource.load_filtered_arrow
    real_load_filtered = ParquetDataResource.load_filtered

    def tracking_load_filtered_arrow(self, filters=None, *, columns=None) -> pa.Table:
        arrow_calls.append(True)
        return real_load_filtered_arrow(self, filters, columns=columns)

    def tracking_load_filtered(self, filters=None, *, columns=None) -> dd.DataFrame:
        dask_calls.append(True)
        return real_load_filtered(self, filters, columns=columns)

    monkeypatch.setattr(ParquetDataResource, "load_filtered_arrow", tracking_load_filtered_arrow)
    monkeypatch.setattr(ParquetDataResource, "load_filtered", tracking_load_filtered)

    with DataGateway(config) as facade:
        frame = facade.load(filters={"status__exact": "active"}, as_pandas=True)

    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [1, 3]
    assert arrow_calls == [True]
    assert not dask_calls


def test_facade_sql_columns_project_lazy_statement(tmp_path) -> None:
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "projected_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))
        description: Mapped[str] = mapped_column(String(128))

    db_path = tmp_path / "projected_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    User(status="active", description="urgent"),
                    User(status="inactive", description="backlog"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        frame = facade.load(
            statement=select(User),
            model=User,
            columns=["id"],
        )

    assert isinstance(frame, dd.DataFrame)
    assert list(frame.columns) == ["id"]
    assert frame.compute()["id"].tolist() == [1, 2]
