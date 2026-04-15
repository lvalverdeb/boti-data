from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import pytest
from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data.distributed import dask_session, describe_client, describe_frame
from boti_data.parquet import ParquetDataConfig
from boti_data.gateway import DataGateway
from boti_data.joins import indexed_left_join, left_join_frames
from boti_data.db import SqlDatabaseConfig


class CaptureLogger:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def debug(self, message: str, *_args, **_kwargs) -> None:
        self.debugs.append(message)

    def info(self, message: str, *_args, **_kwargs) -> None:
        self.infos.append(message)

    def warning(self, message: str, *_args, **_kwargs) -> None:
        self.warnings.append(message)

    def error(self, message: str, *_args, **_kwargs) -> None:
        self.errors.append(message)


def test_dask_session_connects_to_scheduler_address():
    distributed = pytest.importorskip("dask.distributed")
    Client = distributed.Client
    LocalCluster = distributed.LocalCluster
    logger = CaptureLogger()

    with LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    ) as cluster, Client(cluster):
        with dask_session(scheduler_address=cluster.scheduler_address, logger=logger) as client:
            summary = describe_client(client)

    assert summary["workers"] == 1
    assert any("Connected Dask client" in message for message in logger.infos)


def test_describe_frame_reports_dask_metrics():
    frame = dd.from_pandas(pd.DataFrame({"id": [1, 2, 3]}), npartitions=2)

    metrics = describe_frame(frame)

    assert metrics["engine"] == "dask"
    assert metrics["npartitions"] == 2
    assert metrics["columns"] == 1
    assert metrics["graph_tasks"] is not None


def test_gateway_partitioned_sql_diagnostics_log_plan_and_completion(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "diagnostic_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    logger = CaptureLogger()
    db_path = tmp_path / "diagnostics.db"
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
        logger=logger,
    )

    with DataGateway(config) as facade:
        frame = facade.load(
            statement=select(User),
            model=User,
            diagnostics=True,
        )

    assert frame.npartitions == 1
    assert any("Partitioned SQL plan" in message for message in logger.infos)
    assert any("Gateway load completed" in message for message in logger.infos)


def test_indexed_left_join_diagnostics_log_metrics_and_guidance():
    logger = CaptureLogger()
    left = dd.from_pandas(
        pd.DataFrame({"id": pd.Series([1, 2, 3], dtype="Int64"), "left_value": ["a", "b", "c"]}),
        npartitions=2,
    )
    right = dd.from_pandas(
        pd.DataFrame({"id": pd.Series([1, 3], dtype="Int64"), "right_value": ["x", "z"]}),
        npartitions=2,
    )

    joined = indexed_left_join(
        left,
        right,
        join_key="id",
        join_schema_map={"id": "Int64"},
        persist=False,
        diagnostics=True,
        logger=logger,
    )

    assert joined.npartitions >= 1
    assert any("persist=False may recompute" in message for message in logger.warnings)
    assert any("completed in" in message for message in logger.infos)


def test_gateway_partitioned_sql_runs_through_scheduler_address(tmp_path):
    distributed = pytest.importorskip("dask.distributed")
    LocalCluster = distributed.LocalCluster

    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "remote_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    class UserProfile(Base):
        __tablename__ = "remote_profiles"

        id: Mapped[int] = mapped_column(primary_key=True)
        tier: Mapped[str] = mapped_column(String(32))

    db_path = tmp_path / "remote_cluster.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    User(id=1, status="active"),
                    User(id=2, status="inactive"),
                    User(id=3, status="active"),
                ]
            )
            session.add_all(
                [
                    UserProfile(id=1, tier="gold"),
                    UserProfile(id=3, tier="standard"),
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
    ) as cluster:
        with dask_session(scheduler_address=cluster.scheduler_address):
            with DataGateway(config) as facade:
                users = facade.load(statement=select(User), model=User)
                profiles = facade.load(statement=select(UserProfile), model=UserProfile)
                joined = indexed_left_join(
                    users,
                    profiles,
                    join_key="id",
                    join_schema_map={"id": "Int64"},
                    persist=True,
                )
                computed = joined.compute().sort_values("id").reset_index(drop=True)

    assert computed.loc[0, "tier"] == "gold"
    assert pd.isna(computed.loc[1, "tier"])
    assert computed.loc[2, "tier"] == "standard"


def test_gateway_parquet_and_join_run_through_scheduler_address(temp_project_root):
    distributed = pytest.importorskip("dask.distributed")
    LocalCluster = distributed.LocalCluster

    users_path = temp_project_root / "warehouse" / "users.parquet"
    users_path.parent.mkdir(parents=True)
    profiles_path = temp_project_root / "warehouse" / "profiles.parquet"
    pd.DataFrame(
        {
            "id": pd.Series([1, 2, 3], dtype="Int64"),
            "status": ["active", "inactive", "active"],
        }
    ).to_parquet(users_path, index=False)
    pd.DataFrame(
        {
            "id": pd.Series([1, 3], dtype="Int64"),
            "tier": ["gold", "standard"],
        }
    ).to_parquet(profiles_path, index=False)

    users_config = ParquetDataConfig(
        project_root=temp_project_root,
        parquet_storage_path=str(users_path.parent),
        parquet_filename="users",
    )
    profiles_config = ParquetDataConfig(
        project_root=temp_project_root,
        parquet_storage_path=str(profiles_path.parent),
        parquet_filename="profiles",
    )

    with LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=False,
        dashboard_address=None,
    ) as cluster:
        with dask_session(scheduler_address=cluster.scheduler_address):
            with DataGateway(users_config) as users_gateway, DataGateway(profiles_config) as profiles_gateway:
                users = users_gateway.load()
                profiles = profiles_gateway.load()
                joined = indexed_left_join(
                    users,
                    profiles,
                    join_key="id",
                    join_schema_map={"id": "Int64"},
                    persist=True,
                )
                computed = joined.compute().sort_values("id").reset_index(drop=True)

    assert computed.loc[0, "tier"] == "gold"
    assert pd.isna(computed.loc[1, "tier"])
    assert computed.loc[2, "tier"] == "standard"


def test_left_join_frames_diagnostics_warn_for_direct_dask_merge():
    logger = CaptureLogger()
    left = dd.from_pandas(
        pd.DataFrame({"id": pd.Series([1, 2, 3], dtype="Int64"), "left_value": ["a", "b", "c"]}),
        npartitions=2,
    )
    right = dd.from_pandas(
        pd.DataFrame({"id": pd.Series([1, 3], dtype="Int64"), "right_value": ["x", "z"]}),
        npartitions=2,
    )

    joined = left_join_frames(
        left,
        right,
        left_on=["id"],
        join_schema_map={"id": "Int64"},
        diagnostics=True,
        logger=logger,
    )

    assert joined.npartitions >= 1
    assert any("direct Dask merge may trigger a full shuffle" in message for message in logger.warnings)


@pytest.mark.asyncio
async def test_gateway_aload_partitioned_sql_diagnostics_log_plan_and_completion(tmp_path):
    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "async_diagnostic_users"

        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(String(32))

    logger = CaptureLogger()
    db_path = tmp_path / "async_diagnostics.db"
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
        logger=logger,
    )

    async with DataGateway(config) as facade:
        frame = await facade.aload(
            statement=select(User),
            model=User,
            diagnostics=True,
        )

    assert frame.npartitions == 1
    assert any("Partitioned SQL plan" in message for message in logger.infos)
    assert any("Gateway load completed" in message for message in logger.infos)


def test_dask_session_can_create_managed_local_cluster():
    pytest.importorskip("dask.distributed")

    with dask_session(cluster_kwargs={"n_workers": 1, "threads_per_worker": 1, "processes": False, "dashboard_address": None}) as client:
        summary = describe_client(client)

    assert summary["workers"] == 1
