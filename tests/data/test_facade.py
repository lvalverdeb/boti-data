"""
Tests for the data gateway.
"""

from __future__ import annotations

import pickle

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

import boti_data.gateway.core as gateway_core
from boti_data.db import SqlDatabaseConfig, SqlDatabaseResource
from boti_data.gateway import DataGateway
from boti_data.parquet import ParquetDataConfig, ParquetDataResource


class StubLogger:
    def debug(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def error(self, *_args, **_kwargs) -> None:
        pass


def _status_values(frame):
    if isinstance(frame, dd.DataFrame):
        return frame.compute()["status"].tolist()
    if isinstance(frame, pd.DataFrame):
        return frame["status"].tolist()
    if isinstance(frame, pa.Table):
        return frame["status"].to_pylist()
    if isinstance(frame, pl.DataFrame):
        return frame["status"].to_list()
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _id_values(frame):
    if isinstance(frame, dd.DataFrame):
        return frame.compute()["id"].tolist()
    if isinstance(frame, pd.DataFrame):
        return frame["id"].tolist()
    if isinstance(frame, pa.Table):
        return frame["id"].to_pylist()
    if isinstance(frame, pl.DataFrame):
        return frame["id"].to_list()
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def test_facade_loads_sql_with_statement_and_filters(tmp_path):
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


def test_facade_can_opt_into_pandas_sql_reads():
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


def test_facade_loads_parquet_with_high_level_filters(temp_project_root):
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


def test_facade_can_opt_into_pandas_parquet_reads(temp_project_root):
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


def test_facade_parquet_columns_push_down_to_reader(temp_project_root, monkeypatch):
    file_path = temp_project_root / "data" / "projected_events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"id": [1, 2, 3], "status": ["active", "inactive", "active"], "description": ["a", "b", "c"]}
    ).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="projected_events",
    )

    seen_columns: list[list[str] | None] = []
    real_load_files = ParquetDataResource.load_files

    def tracking_load_files(self, filters=None, *, columns=None):
        seen_columns.append(columns)
        return real_load_files(self, filters, columns=columns)

    monkeypatch.setattr(ParquetDataResource, "load_files", tracking_load_files)

    with DataGateway(config) as facade:
        frame = facade.load(filters={"status__exact": "active"}, columns=["id"])

    assert isinstance(frame, dd.DataFrame)
    assert list(frame.columns) == ["id"]
    assert frame.compute()["id"].tolist() == [1, 3]
    assert seen_columns == [["id"]]


def test_facade_pandas_parquet_reads_use_arrow_eager_path(temp_project_root, monkeypatch):
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

    def tracking_load_filtered_arrow(self, filters=None, *, columns=None):
        arrow_calls.append(True)
        return real_load_filtered_arrow(self, filters, columns=columns)

    def tracking_load_filtered(self, filters=None, *, columns=None):
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


def test_facade_can_return_arrow_table_for_sql(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "arrow_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "arrow_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
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
            return_type="arrow",
        )

    assert isinstance(frame, pa.Table)
    assert frame.column_names == ["id", "status"]
    assert frame.num_rows == 2


def test_facade_arrow_sql_uses_eager_fetch_by_default(tmp_path, monkeypatch):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "arrow_sql_fetch_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "arrow_sql_fetch.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    eager_calls: list[str] = []
    lazy_calls: list[bool] = []
    real_load_sql = gateway_core.load_sql
    real_load_sql_partitioned = gateway_core.load_sql_partitioned

    def tracking_load_sql(resource, request):
        eager_calls.append(request.return_type)
        return real_load_sql(resource, request)

    def tracking_load_sql_partitioned(config, resource, request):
        lazy_calls.append(True)
        return real_load_sql_partitioned(config, resource, request)

    monkeypatch.setattr(gateway_core, "load_sql", tracking_load_sql)
    monkeypatch.setattr(gateway_core, "load_sql_partitioned", tracking_load_sql_partitioned)

    with DataGateway(config) as facade:
        frame = facade.load(
            statement=select(User),
            model=User,
            return_type="arrow",
        )

    assert isinstance(frame, pa.Table)
    assert eager_calls == ["arrow"]
    assert not lazy_calls


def test_facade_can_return_arrow_table_for_parquet(temp_project_root):
    file_path = temp_project_root / "data" / "arrow_events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "status": ["active", "inactive", "active"]}).to_parquet(
        file_path, index=False
    )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="arrow_events",
    )

    with DataGateway(config) as facade:
        frame = facade.load(filters={"status__exact": "active"}, return_type="arrow")

    assert isinstance(frame, pa.Table)
    assert frame.column_names == ["id", "status"]
    assert frame.num_rows == 2


def test_facade_can_return_polars_frame_for_sql(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "polars_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "polars_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
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
            return_type="polars",
        )

    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ["id", "status"]
    assert frame["status"].to_list() == ["active", "inactive"]


def test_facade_polars_sql_uses_arrow_fetch(tmp_path, monkeypatch):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "polars_sql_fetch_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "polars_sql_fetch.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    eager_calls: list[str] = []
    real_load_sql = gateway_core.load_sql

    def tracking_load_sql(resource, request):
        eager_calls.append(request.return_type)
        return real_load_sql(resource, request)

    monkeypatch.setattr(gateway_core, "load_sql", tracking_load_sql)

    with DataGateway(config) as facade:
        frame = facade.load(
            statement=select(User),
            model=User,
            return_type="polars",
        )

    assert isinstance(frame, pl.DataFrame)
    assert eager_calls == ["arrow"]


def test_facade_can_force_lazy_fetch_for_pandas_sql(tmp_path, monkeypatch):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "lazy_pandas_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "lazy_pandas_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    eager_calls: list[bool] = []
    lazy_calls: list[bool] = []
    real_load_sql = gateway_core.load_sql
    real_load_sql_partitioned = gateway_core.load_sql_partitioned

    def tracking_load_sql(resource, request):
        eager_calls.append(True)
        return real_load_sql(resource, request)

    def tracking_load_sql_partitioned(config, resource, request):
        lazy_calls.append(True)
        return real_load_sql_partitioned(config, resource, request)

    monkeypatch.setattr(gateway_core, "load_sql", tracking_load_sql)
    monkeypatch.setattr(gateway_core, "load_sql_partitioned", tracking_load_sql_partitioned)

    with DataGateway(config) as facade:
        frame = facade.load(
            statement=select(User),
            model=User,
            return_type="pandas",
            execution_mode="lazy",
        )

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active", "inactive"]
    assert lazy_calls == [True]
    assert not eager_calls


def test_facade_resilient_persist_uses_safe_persist_for_lazy_sql(tmp_path, monkeypatch):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "resilient_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "resilient_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    calls: list[int] = []

    def tracking_safe_persist(frame, *, dask_client=None, logger=None):
        calls.append(frame.npartitions)
        return frame.persist()

    monkeypatch.setattr(gateway_core, "safe_persist", tracking_safe_persist)

    with DataGateway(config) as facade:
        frame = facade.load(
            statement=select(User),
            model=User,
            persist=True,
            resilient=True,
        )

    assert isinstance(frame, dd.DataFrame)
    assert calls == [frame.npartitions]
    assert frame.compute()["status"].tolist() == ["active", "inactive"]


def test_facade_dry_run_returns_lazy_dask_graph_without_persist(tmp_path, monkeypatch):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "dry_run_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "dry_run_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    def fail_safe_persist(*_args, **_kwargs):
        raise AssertionError("safe_persist should not be called during dry run")

    monkeypatch.setattr(gateway_core, "safe_persist", fail_safe_persist)

    with DataGateway(config) as facade:
        frame = facade.load(
            statement=select(User),
            model=User,
            persist=True,
            resilient=True,
            dry_run=True,
        )
        computed = frame.compute()

    assert isinstance(frame, dd.DataFrame)
    assert computed["status"].tolist() == ["active", "inactive"]


def test_facade_dry_run_rejects_eager_return_types(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "dry_run_eager_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "dry_run_eager_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(User(status="active"))
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        with pytest.raises(ValueError, match="dry_run=True is only supported"):
            facade.load(
                statement=select(User),
                model=User,
                return_type="pandas",
                dry_run=True,
            )


def test_facade_preview_uses_safe_head_for_lazy_sql(tmp_path, monkeypatch):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "preview_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "preview_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    calls: list[tuple[int, int]] = []

    def tracking_safe_head(frame, *, n=5, npartitions=1, dask_client=None, logger=None, dry_run=False):
        calls.append((n, npartitions))
        return frame.head(n, npartitions=npartitions)

    monkeypatch.setattr(gateway_core, "safe_head", tracking_safe_head)

    with DataGateway(config) as facade:
        preview = facade.preview(
            statement=select(User),
            model=User,
            n=1,
            npartitions=1,
        )

    assert preview["status"].tolist() == ["active"]
    assert calls == [(1, 1)]


@pytest.mark.asyncio
async def test_facade_aload_resilient_persist_uses_safe_persist_for_lazy_sql(tmp_path, monkeypatch):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "async_resilient_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "async_resilient_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    calls: list[int] = []

    def tracking_safe_persist(frame, *, dask_client=None, logger=None):
        calls.append(frame.npartitions)
        return frame.persist()

    monkeypatch.setattr(gateway_core, "safe_persist", tracking_safe_persist)

    async with DataGateway(config) as facade:
        frame = await facade.aload(
            statement=select(User),
            model=User,
            persist=True,
            resilient=True,
        )

    assert isinstance(frame, dd.DataFrame)
    assert calls == [frame.npartitions]
    assert frame.compute()["status"].tolist() == ["active", "inactive"]


@pytest.mark.asyncio
async def test_facade_apreview_uses_async_safe_head_for_lazy_sql(tmp_path, monkeypatch):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "apreview_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "apreview_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    calls: list[tuple[int, int]] = []

    async def tracking_async_safe_head(
        frame,
        *,
        n=5,
        npartitions=1,
        dask_client=None,
        logger=None,
        dry_run=False,
    ):
        calls.append((n, npartitions))
        return frame.head(n, npartitions=npartitions)

    monkeypatch.setattr(gateway_core, "async_safe_head", tracking_async_safe_head)

    async with DataGateway(config) as facade:
        preview = await facade.apreview(
            statement=select(User),
            model=User,
            n=1,
            npartitions=1,
        )

    assert preview["status"].tolist() == ["active"]
    assert calls == [(1, 1)]


def test_facade_can_return_polars_frame_for_parquet(temp_project_root):
    file_path = temp_project_root / "data" / "polars_events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "status": ["active", "inactive", "active"]}).to_parquet(
        file_path, index=False
    )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="polars_events",
    )

    with DataGateway(config) as facade:
        frame = facade.load(filters={"status__exact": "active"}, return_type="polars")

    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ["id", "status"]
    assert frame["id"].to_list() == [1, 3]


@pytest.mark.parametrize(
    ("return_type", "expected_type"),
    [
        ("dask", dd.DataFrame),
        ("pandas", pd.DataFrame),
        ("arrow", pa.Table),
        ("polars", pl.DataFrame),
        ("auto", pd.DataFrame),
    ],
)
def test_facade_structured_sql_return_type_matrix(tmp_path, return_type, expected_type):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "matrix_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "matrix_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        frame = facade.load(statement=select(User), model=User, return_type=return_type)

    assert isinstance(frame, expected_type)
    assert _status_values(frame) == ["active", "inactive"]


@pytest.mark.parametrize(
    ("return_type", "expected_type"),
    [
        ("dask", dd.DataFrame),
        ("pandas", pd.DataFrame),
        ("arrow", pa.Table),
        ("polars", pl.DataFrame),
        ("auto", pd.DataFrame),
    ],
)
def test_facade_parquet_return_type_matrix(temp_project_root, return_type, expected_type):
    file_path = temp_project_root / "data" / "matrix_events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "status": ["active", "inactive", "active"]}).to_parquet(
        file_path, index=False
    )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="matrix_events",
    )

    with DataGateway(config) as facade:
        frame = facade.load(filters={"status__exact": "active"}, return_type=return_type)

    assert isinstance(frame, expected_type)
    assert _id_values(frame) == [1, 3]


def test_facade_auto_return_type_prefers_pandas_for_small_sql(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "auto_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "auto_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        frame = facade.load(statement=select(User), model=User, return_type="auto")

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active", "inactive"]


def test_facade_auto_small_sql_avoids_full_plan(tmp_path, monkeypatch):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "auto_probe_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "auto_probe_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    def fail_plan_request(self, request):
        raise AssertionError("full plan_request should not be used for auto SQL sizing")

    monkeypatch.setattr(gateway_core.SqlPartitionPlanner, "plan_request", fail_plan_request)

    with DataGateway(config) as facade:
        frame = facade.load(statement=select(User), model=User, return_type="auto")

    assert isinstance(frame, pd.DataFrame)


def test_facade_auto_return_type_uses_dask_for_large_sql(tmp_path, monkeypatch):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "auto_large_sql_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "auto_large_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(status="active"), User(status="inactive")])
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )
    monkeypatch.setattr(gateway_core, "_AUTO_EAGER_MAX_ROWS", 1)

    with DataGateway(config) as facade:
        frame = facade.load(statement=select(User), model=User, return_type="auto")

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["status"].tolist() == ["active", "inactive"]


def test_facade_auto_return_type_prefers_pandas_for_small_parquet(temp_project_root):
    file_path = temp_project_root / "data" / "auto_events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "status": ["active", "inactive", "active"]}).to_parquet(
        file_path, index=False
    )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="auto_events",
    )

    with DataGateway(config) as facade:
        frame = facade.load(filters={"status__exact": "active"}, return_type="auto")

    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [1, 3]


def test_facade_auto_parquet_resolves_files_once(temp_project_root, monkeypatch):
    file_path = temp_project_root / "data" / "auto_once_events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="auto_once_events",
    )

    calls: list[bool] = []
    real_resolve_files = ParquetDataResource._resolve_files_to_load

    def tracking_resolve_files(self):
        calls.append(True)
        return real_resolve_files(self)

    monkeypatch.setattr(ParquetDataResource, "_resolve_files_to_load", tracking_resolve_files)

    with DataGateway(config) as facade:
        resolved = facade._resolve_auto_parquet_return_type({})

    assert resolved == "pandas"
    assert calls == [True]


def test_facade_auto_return_type_uses_dask_for_large_parquet(temp_project_root, monkeypatch):
    file_path = temp_project_root / "data" / "auto_large_events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "status": ["active", "inactive", "active"]}).to_parquet(
        file_path, index=False
    )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="auto_large_events",
    )
    monkeypatch.setattr(gateway_core, "_AUTO_EAGER_MAX_BYTES", 1)

    with DataGateway(config) as facade:
        frame = facade.load(filters={"status__exact": "active"}, return_type="auto")

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1, 3]


def test_facade_sql_columns_project_lazy_statement(tmp_path):
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


def test_facade_loads_partitioned_sql_with_distributed_client(tmp_path):
    distributed = pytest.importorskip("dask.distributed")
    Client = distributed.Client
    LocalCluster = distributed.LocalCluster

    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "distributed_facade_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "distributed_facade.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    User(id=1, status="active"),
                    User(id=2, status="active"),
                    User(id=3, status="inactive"),
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

    with LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    ) as cluster, Client(cluster):
        with DataGateway(config) as facade:
            frame = facade.load(
                statement=select(User),
                model=User,
                chunk_size=1,
                persist=True,
            )

        assert isinstance(frame, dd.DataFrame)
        assert frame.compute()["status"].tolist() == ["active", "active", "inactive"]


def test_sql_gateway_is_pickleable_for_distributed_use(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "pickleable_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "pickleable_gateway.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    User(status="active"),
                    User(status="inactive"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        allow_pickle=True,
    )

    facade = DataGateway(
        config,
        table="pickleable_users",
        sticky_filters={"status": "active"},
    )
    restored = None
    try:
        with SqlDatabaseResource.trusted_unpickle_scope():
            restored = pickle.loads(pickle.dumps(facade))

        frame = restored.load(as_pandas=True)
    finally:
        facade.close()
        if restored is not None:
            restored.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active"]


def test_parquet_gateway_is_pickleable_for_distributed_use(temp_project_root):
    file_path = temp_project_root / "distributed" / "gateway_users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"id": [1, 2], "status": ["active", "inactive"]}
    ).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="gateway_users",
        allow_pickle=True,
    )

    facade = DataGateway(config)
    restored = None
    try:
        with ParquetDataResource.trusted_unpickle_scope():
            restored = pickle.loads(pickle.dumps(facade))

        frame = restored.load(filters={"status__exact": "active"}, as_pandas=True)
    finally:
        facade.close()
        if restored is not None:
            restored.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [1]


def test_facade_rejects_lazy_parquet_limit(temp_project_root):
    file_path = temp_project_root / "data" / "limited_events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3]}).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="limited_events",
    )

    with DataGateway(config) as facade:
        with pytest.raises(ValueError, match="Lazy parquet gateway loads do not support exact limit"):
            facade.load(limit=2)


def test_facade_from_backend_preserves_strict_validation():
    with pytest.raises(Exception):
        DataGateway.from_backend("parquet", parquet_storage_path="/tmp/data", unexpected=True)


def test_facade_accepts_singleton_tuple_sql_config(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "tuple_config_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "tuple_config_users.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(User(status="active"))
            session.commit()
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    with DataGateway((config,), table="tuple_config_users") as facade:
        frame = facade.load(as_pandas=True)

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active"]


@pytest.mark.asyncio
async def test_facade_aloads_parquet(temp_project_root):
    file_path = temp_project_root / "data" / "async_users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"id": [1, 2], "status": ["active", "inactive"], "description": ["urgent", "routine"]}
    ).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="async_users",
    )

    async with DataGateway(config) as facade:
        frame = await facade.aload(filters={"status__exact": "active"})

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1]


@pytest.mark.asyncio
async def test_facade_aloads_sql_with_async_resource(monkeypatch):
    class FakeAsyncConnection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def run_sync(self, fn):
            engine = create_engine("sqlite:///:memory:")
            metadata = MetaData()
            table = Table(
                "users",
                metadata,
                Column("id", Integer, primary_key=True),
                Column("status", String),
            )
            metadata.create_all(engine)
            with engine.begin() as conn:
                conn.execute(table.insert(), [{"id": 1, "status": "active"}])
            with engine.connect() as conn:
                return fn(conn)

    class FakeAsyncEngine:
        def connect(self):
            return FakeAsyncConnection()

    class FakeAsyncSqlDatabaseResource:
        def __init__(self, _config):
            self.logger = StubLogger()
            self.engine = FakeAsyncEngine()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    monkeypatch.setattr("boti_data.gateway.core.AsyncSqlDatabaseResource", FakeAsyncSqlDatabaseResource)

    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    async with DataGateway(config) as facade:
        frame = await facade.aload(sql="SELECT id, status FROM users", as_pandas=True)

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active"]


def test_facade_rejects_lazy_raw_sql_without_statement_model():
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        with pytest.raises(ValueError, match="Lazy SQL gateway loads require a SQLAlchemy Select statement and model"):
            facade.load(sql="SELECT 1 AS id")


def test_facade_loads_partitioned_sql_as_lazy_dask(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "partitioned_facade_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "facade_partitioned.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    User(id=1, status="active"),
                    User(id=2, status="active"),
                    User(id=3, status="inactive"),
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
            partitioned=True,
            chunk_size=2,
        )

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1, 2, 3]


@pytest.mark.asyncio
async def test_facade_aloads_partitioned_sql_with_sync_dsn(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "async_partitioned_facade_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "async_facade_partitioned.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    User(id=1, status="active"),
                    User(id=2, status="inactive"),
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

    async with DataGateway(config) as facade:
        frame = await facade.aload(
            statement=select(User),
            model=User,
            chunk_size=1,
        )

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["status"].tolist() == ["active", "inactive"]
