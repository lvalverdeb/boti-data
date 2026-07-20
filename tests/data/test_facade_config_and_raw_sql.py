"""
Data gateway tests: config/backend edge cases (strict validation, tuple
configs, lazy parquet limits), async parquet/SQL loads, and the raw-SQL
opt-in policy guardrails.

Split out of test_facade.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

from typing import Any

import dask.dataframe as dd
import pandas as pd
import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine
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


def test_facade_rejects_lazy_parquet_limit(temp_project_root) -> None:
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
        with pytest.raises(
            ValueError, match="Lazy parquet gateway loads do not support exact limit"
        ):
            facade.load(limit=2)


def test_facade_from_backend_preserves_strict_validation() -> None:
    with pytest.raises(Exception):
        DataGateway.from_backend("parquet", parquet_storage_path="/tmp/data", unexpected=True)


def test_facade_accepts_singleton_tuple_sql_config(tmp_path) -> None:
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
async def test_facade_aloads_parquet(temp_project_root) -> None:
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


class _FakeAsyncConnection:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def run_sync(self, fn) -> Any:
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


class _FakeAsyncEngine:
    def connect(self) -> _FakeAsyncConnection:
        return _FakeAsyncConnection()


class _FakeAsyncSqlDatabaseResource:
    def __init__(self, _config):
        self.logger = StubLogger()
        self.engine = _FakeAsyncEngine()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.mark.asyncio
async def test_facade_aloads_sql_with_async_resource(monkeypatch) -> None:
    monkeypatch.setattr(
        "boti_data.gateway._backend_strategies.AsyncSqlDatabaseResource",
        _FakeAsyncSqlDatabaseResource,
    )

    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    async with DataGateway(config) as facade:
        frame = await facade.aload(
            sql="SELECT id, status FROM users",
            as_pandas=True,
            allow_raw_sql=True,
        )

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active"]


def test_facade_rejects_lazy_raw_sql_without_statement_model() -> None:
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        with pytest.raises(
            ValueError,
            match="Lazy SQL gateway loads require a SQLAlchemy Select statement and model",
        ):
            facade.load(sql="SELECT 1 AS id", allow_raw_sql=True)


# test_facade_rejects_raw_sql_by_default_even_for_eager_reads was identical to
# tests/security/test_regressions_raw_sql_and_datacube.py::test_raw_sql_requires_explicit_allow_flag;
# that's the authoritative copy (dedicated security-regression suite).


def test_facade_rejects_mutating_raw_sql_even_when_opted_in() -> None:
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        with pytest.raises(ValueError, match="only supports single-statement read-only"):
            facade.load(sql="DELETE FROM users", as_pandas=True, allow_raw_sql=True)


def test_facade_rejects_multi_statement_raw_sql_even_when_opted_in() -> None:
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        with pytest.raises(ValueError, match="only supports single-statement read-only"):
            facade.load(
                sql="SELECT 1 AS id; DROP TABLE users",
                as_pandas=True,
                allow_raw_sql=True,
            )


# test_facade_raw_sql_policy_disabled_blocks_even_explicit_opt_in was identical
# to tests/security/test_regressions_raw_sql_and_datacube.py::test_raw_sql_policy_disabled_rejects_raw_sql_even_with_allow_flag;
# that's the authoritative copy (dedicated security-regression suite).
