"""
Data gateway tests: lazy-SQL lifecycle behavior around persist/resilient
loads, dry-run semantics, and preview (sync and async variants).

Split out of test_facade.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pytest
from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

import boti_data.gateway.core as gateway_core
import boti_data.gateway.post_process as gateway_post_process
from boti_data.db import SqlDatabaseConfig
from boti_data.gateway import DataGateway


class _LazyPandasSqlBase(DeclarativeBase):
    pass


class _LazyPandasSqlUser(_LazyPandasSqlBase):
    __tablename__ = "lazy_pandas_sql_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(32))


def _seed_lazy_pandas_sql_db(tmp_path) -> SqlDatabaseConfig:
    db_path = tmp_path / "lazy_pandas_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        _LazyPandasSqlBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [_LazyPandasSqlUser(status="active"), _LazyPandasSqlUser(status="inactive")]
            )
            session.commit()
    finally:
        engine.dispose()

    return SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )


def test_facade_can_force_lazy_fetch_for_pandas_sql(tmp_path, monkeypatch) -> None:
    config = _seed_lazy_pandas_sql_db(tmp_path)
    eager_calls: list[bool] = []
    lazy_calls: list[bool] = []
    from boti_data.gateway.loaders import load_sql as real_load_sql
    from boti_data.gateway.loaders import load_sql_partitioned as real_load_sql_partitioned

    def tracking_load_sql(resource, request) -> pd.DataFrame | pa.Table:
        eager_calls.append(True)
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
            statement=select(_LazyPandasSqlUser),
            model=_LazyPandasSqlUser,
            return_type="pandas",
            execution_mode="lazy",
        )

    assert isinstance(frame, pd.DataFrame)
    assert frame["status"].tolist() == ["active", "inactive"]
    assert lazy_calls == [True]
    assert not eager_calls


def test_facade_resilient_persist_uses_safe_persist_for_lazy_sql(tmp_path, monkeypatch) -> None:
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

    def tracking_safe_persist(frame, *, dask_client=None, logger=None) -> dd.DataFrame:
        calls.append(frame.npartitions)
        return frame.persist()

    monkeypatch.setattr(gateway_post_process, "safe_persist", tracking_safe_persist)

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


def test_facade_dry_run_returns_lazy_dask_graph_without_persist(tmp_path, monkeypatch) -> None:
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

    def fail_safe_persist(*_args, **_kwargs) -> None:
        raise AssertionError("safe_persist should not be called during dry run")

    monkeypatch.setattr(gateway_post_process, "safe_persist", fail_safe_persist)

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


def test_facade_dry_run_rejects_eager_return_types(tmp_path) -> None:
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


def test_facade_preview_uses_safe_head_for_lazy_sql(tmp_path, monkeypatch) -> None:
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

    # Not a copy-pasted twin: this sync tracker and apreview's async tracker
    # below monkeypatch two genuinely different targets (safe_head vs
    # async_safe_head); the bodies match because both just record args and
    # delegate to frame.head(), not because of duplication.
    # spaghetti-ignore[duplicate-function-body]: see above
    def tracking_safe_head(
        frame, *, n=5, npartitions=1, dask_client=None, logger=None, dry_run=False
    ) -> pd.DataFrame:
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
async def test_facade_aload_resilient_persist_uses_safe_persist_for_lazy_sql(
    tmp_path, monkeypatch
) -> None:
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

    def tracking_safe_persist(frame, *, dask_client=None, logger=None) -> dd.DataFrame:
        calls.append(frame.npartitions)
        return frame.persist()

    monkeypatch.setattr(gateway_post_process, "safe_persist", tracking_safe_persist)

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


class _ApreviewSqlBase(DeclarativeBase):
    pass


class _ApreviewSqlUser(_ApreviewSqlBase):
    __tablename__ = "apreview_sql_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(32))


def _seed_apreview_sql_db(tmp_path) -> SqlDatabaseConfig:
    db_path = tmp_path / "apreview_sql.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        _ApreviewSqlBase.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [_ApreviewSqlUser(status="active"), _ApreviewSqlUser(status="inactive")]
            )
            session.commit()
    finally:
        engine.dispose()

    return SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )


@pytest.mark.asyncio
async def test_facade_apreview_uses_async_safe_head_for_lazy_sql(tmp_path, monkeypatch) -> None:
    config = _seed_apreview_sql_db(tmp_path)
    calls: list[tuple[int, int]] = []

    async def tracking_async_safe_head(
        frame,
        *,
        n=5,
        npartitions=1,
        dask_client=None,
        logger=None,
        dry_run=False,
    ) -> pd.DataFrame:
        calls.append((n, npartitions))
        return frame.head(n, npartitions=npartitions)

    monkeypatch.setattr(gateway_core, "async_safe_head", tracking_async_safe_head)

    async with DataGateway(config) as facade:
        preview = await facade.apreview(
            statement=select(_ApreviewSqlUser),
            model=_ApreviewSqlUser,
            n=1,
            npartitions=1,
        )

    assert preview["status"].tolist() == ["active"]
    assert calls == [(1, 1)]
