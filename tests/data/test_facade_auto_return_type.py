"""
Data gateway tests: "auto" return-type sizing heuristics (SQL and parquet)
and the related IN-list auto-chunking behavior.

Split out of test_facade.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

import boti_data.gateway.return_type as gateway_return_type
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


def test_facade_auto_return_type_prefers_pandas_for_small_sql(tmp_path) -> None:
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


def test_facade_auto_small_sql_avoids_full_plan(tmp_path, monkeypatch) -> None:
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

    def fail_plan_request(self, request) -> None:
        raise AssertionError("full plan_request should not be used for auto SQL sizing")

    from boti_data.db.partitioned_planner import SqlPartitionPlanner

    monkeypatch.setattr(SqlPartitionPlanner, "plan_request", fail_plan_request)

    with DataGateway(config) as facade:
        frame = facade.load(statement=select(User), model=User, return_type="auto")

    assert isinstance(frame, pd.DataFrame)


def test_facade_auto_return_type_uses_dask_for_large_sql(tmp_path, monkeypatch) -> None:
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
    monkeypatch.setattr(gateway_return_type, "_AUTO_EAGER_MAX_ROWS", 1)

    with DataGateway(config) as facade:
        frame = facade.load(statement=select(User), model=User, return_type="auto")

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["status"].tolist() == ["active", "inactive"]


def test_facade_auto_return_type_prefers_pandas_for_small_parquet(temp_project_root) -> None:
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


def test_facade_auto_parquet_resolves_files_once(temp_project_root, monkeypatch) -> None:
    file_path = temp_project_root / "data" / "auto_once_events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(
        file_path, index=False
    )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="auto_once_events",
    )

    calls: list[bool] = []
    real_resolve_files = ParquetDataResource._resolve_files_to_load

    def tracking_resolve_files(self) -> list[str]:
        calls.append(True)
        return real_resolve_files(self)

    monkeypatch.setattr(ParquetDataResource, "_resolve_files_to_load", tracking_resolve_files)

    with DataGateway(config) as facade:
        resolved = facade._resolve_auto_return_type({"backend": "parquet"})

    assert resolved == "pandas"
    assert len(calls) == 1


def test_facade_auto_return_type_uses_dask_for_large_parquet(
    temp_project_root, monkeypatch
) -> None:
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
    monkeypatch.setattr(gateway_return_type, "_AUTO_EAGER_MAX_BYTES", 1)
    import boti_data.gateway._backend_strategies as _gw_strategies

    monkeypatch.setattr(_gw_strategies, "_AUTO_EAGER_MAX_BYTES", 1)

    with DataGateway(config) as facade:
        frame = facade.load(filters={"status__exact": "active"}, return_type="auto")

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1, 3]


def test_facade_auto_in_chunking_disables_eager_sql_for_medium_in_lists() -> None:
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        size, concurrency = facade._resolve_in_chunk_controls(
            {"id__in": list(range(5_000))},
            execution_mode="eager",
            strategy="auto",
            in_chunk_size_raw=None,
            in_chunk_concurrency_raw=None,
        )

    assert size == 0
    assert concurrency is None


def test_facade_auto_in_chunking_keeps_lazy_sql_chunking_for_medium_in_lists() -> None:
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        size, concurrency = facade._resolve_in_chunk_controls(
            {"id__in": list(range(5_000))},
            execution_mode="lazy",
            strategy="auto",
            in_chunk_size_raw=None,
            in_chunk_concurrency_raw=None,
        )

    assert size == 900
    assert concurrency is not None and concurrency >= 1


def test_facade_auto_in_chunking_disables_eager_sql_for_ten_thousand_sqlite_values() -> None:
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        size, concurrency = facade._resolve_in_chunk_controls(
            {"id__in": list(range(10_000))},
            execution_mode="eager",
            strategy="auto",
            in_chunk_size_raw=None,
            in_chunk_concurrency_raw=None,
        )

    assert size == 0
    assert concurrency is None


def test_facade_in_chunk_diagnostics_logs_sync_fanout(tmp_path) -> None:
    class CaptureLogger:
        def __init__(self) -> None:
            self.infos: list[str] = []

        def info(self, message: str, *_args, **_kwargs) -> None:
            self.infos.append(message)

    class Base(DeclarativeBase):
        pass

    class Event(Base):
        __tablename__ = "chunk_diag_events"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "chunk_diag.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([Event(id=i, status="ok") for i in range(1, 501)])
            session.commit()
    finally:
        engine.dispose()

    logger = CaptureLogger()
    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        logger=logger,
    )

    with DataGateway(config, table="chunk_diag_events") as facade:
        frame = facade.load(
            id__in=list(range(1, 401)),
            return_type="pandas",
            execution_mode="eager",
            diagnostics=True,
            in_chunk_size=100,
            in_chunk_concurrency=1,
        )

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 400
    assert any("Gateway IN chunk fan-out sync" in message for message in logger.infos)
