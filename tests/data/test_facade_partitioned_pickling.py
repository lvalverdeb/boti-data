"""
Data gateway tests: partitioned SQL loads (sync/async, distributed client),
and gateway pickling for distributed use (SQL and parquet backends).

Split out of test_facade.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import pickle

import dask.dataframe as dd
import pandas as pd
import pytest
from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

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


class _DistributedFacadeBase(DeclarativeBase):
    pass


class _DistributedFacadeUser(_DistributedFacadeBase):
    __tablename__ = "distributed_facade_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(32))


def _seed_distributed_facade_db(tmp_path) -> SqlDatabaseConfig:
    db_path = tmp_path / "distributed_facade.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        _DistributedFacadeBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    _DistributedFacadeUser(id=1, status="active"),
                    _DistributedFacadeUser(id=2, status="active"),
                    _DistributedFacadeUser(id=3, status="inactive"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    return SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )


def test_facade_loads_partitioned_sql_with_distributed_client(tmp_path) -> None:
    distributed = pytest.importorskip("dask.distributed")
    Client = distributed.Client
    LocalCluster = distributed.LocalCluster

    config = _seed_distributed_facade_db(tmp_path)

    with (
        LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=False,
            dashboard_address=":0",
        ) as cluster,
        Client(cluster),
    ):
        with DataGateway(config) as facade:
            frame = facade.load(
                statement=select(_DistributedFacadeUser),
                model=_DistributedFacadeUser,
                chunk_size=1,
                persist=True,
            )

        assert isinstance(frame, dd.DataFrame)
        assert frame.compute()["status"].tolist() == ["active", "active", "inactive"]


def test_sql_gateway_is_pickleable_for_distributed_use(tmp_path) -> None:
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


def test_parquet_gateway_is_pickleable_for_distributed_use(temp_project_root) -> None:
    file_path = temp_project_root / "distributed" / "gateway_users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(
        file_path, index=False
    )

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


def test_facade_loads_partitioned_sql_as_lazy_dask(tmp_path) -> None:
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
async def test_facade_aloads_partitioned_sql_with_sync_dsn(tmp_path) -> None:
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
