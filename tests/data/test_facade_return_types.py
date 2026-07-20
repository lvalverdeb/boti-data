"""
Data gateway tests: arrow/polars return types for SQL and parquet backends,
plus the full return-type matrix (dask/pandas/arrow/polars/auto).

Split out of test_facade.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data.db import SqlDatabaseConfig
from boti_data.gateway import DataGateway
from boti_data.parquet import ParquetDataConfig


class StubLogger:
    def debug(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def error(self, *_args, **_kwargs) -> None:
        pass


def _extract_column_values(frame: Any, column: str) -> list:
    extractors: list[tuple[type, Callable[[Any], list]]] = [
        (dd.DataFrame, lambda frame: frame.compute()[column].tolist()),
        (pd.DataFrame, lambda frame: frame[column].tolist()),
        (pa.Table, lambda frame: frame[column].to_pylist()),
        (pl.DataFrame, lambda frame: frame[column].to_list()),
    ]
    for frame_type, extract in extractors:
        if isinstance(frame, frame_type):
            return extract(frame)
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _status_values(frame):
    return _extract_column_values(frame, "status")


def _id_values(frame):
    return _extract_column_values(frame, "id")


def test_facade_can_return_arrow_table_for_sql(tmp_path) -> None:
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


class _ArrowSqlFetchBase(DeclarativeBase):
    pass


class _ArrowSqlFetchUser(_ArrowSqlFetchBase):
    __tablename__ = "arrow_sql_fetch_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(32))


def _seed_arrow_sql_fetch_db(tmp_path) -> SqlDatabaseConfig:
    db_path = tmp_path / "arrow_sql_fetch.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        _ArrowSqlFetchBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [_ArrowSqlFetchUser(status="active"), _ArrowSqlFetchUser(status="inactive")]
            )
            session.commit()
    finally:
        engine.dispose()

    return SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )


def test_facade_arrow_sql_uses_eager_fetch_by_default(tmp_path, monkeypatch) -> None:
    config = _seed_arrow_sql_fetch_db(tmp_path)
    eager_calls: list[str] = []
    lazy_calls: list[bool] = []
    from boti_data.gateway.loaders import load_sql as real_load_sql
    from boti_data.gateway.loaders import load_sql_partitioned as real_load_sql_partitioned

    def tracking_load_sql(resource, request) -> pd.DataFrame | pa.Table:
        eager_calls.append(request.return_type)
        return real_load_sql(resource, request)

    def tracking_load_sql_partitioned(config, resource, request) -> pd.DataFrame | dd.DataFrame:
        lazy_calls.append(True)
        return real_load_sql_partitioned(config, resource, request)

    monkeypatch.setattr("boti_data.gateway._backend_strategies.load_sql", tracking_load_sql)
    monkeypatch.setattr(
        "boti_data.gateway._backend_strategies.load_sql_partitioned", tracking_load_sql_partitioned
    )

    with DataGateway(config) as facade:
        frame = facade.load(
            statement=select(_ArrowSqlFetchUser),
            model=_ArrowSqlFetchUser,
            return_type="arrow",
        )

    assert isinstance(frame, pa.Table)
    assert eager_calls == ["arrow"]
    assert not lazy_calls


def test_facade_can_return_arrow_table_for_parquet(temp_project_root) -> None:
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


def test_facade_can_return_polars_frame_for_sql(tmp_path) -> None:
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


def test_facade_polars_sql_uses_arrow_fetch(tmp_path, monkeypatch) -> None:
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
    from boti_data.gateway.loaders import load_sql as _real_load_sql

    real_load_sql = _real_load_sql

    def tracking_load_sql(resource, request) -> pd.DataFrame | pa.Table:
        eager_calls.append(request.return_type)
        return real_load_sql(resource, request)

    monkeypatch.setattr("boti_data.gateway._backend_strategies.load_sql", tracking_load_sql)

    with DataGateway(config) as facade:
        frame = facade.load(
            statement=select(User),
            model=User,
            return_type="polars",
        )

    assert isinstance(frame, pl.DataFrame)
    assert eager_calls == ["arrow"]


def test_facade_can_return_polars_frame_for_parquet(temp_project_root) -> None:
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
def test_facade_structured_sql_return_type_matrix(tmp_path, return_type, expected_type) -> None:
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
def test_facade_parquet_return_type_matrix(temp_project_root, return_type, expected_type) -> None:
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
