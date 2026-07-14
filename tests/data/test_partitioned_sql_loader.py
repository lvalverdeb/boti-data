"""
Tests for the partition-aware SQL -> Dask loader.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import dask.dataframe as dd
import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import Date, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data.db import (
    SqlDatabaseConfig,
    SqlPartitionedLoader,
    SqlPartitionedLoadRequest,
)
from boti_data.db.partitioned_execution import SqlPartitionExecutor
from boti_data.db.partitioned_planner import SqlPartitionPlanner
from boti_data.db.partitioned_types import MAX_PARTITION_FETCH_CONCURRENCY


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "partitioned_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(String(128))


class Event(Base):
    __tablename__ = "partitioned_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date)


def _create_user_db(tmp_path, rows: list[dict[str, object]]) -> SqlDatabaseConfig:
    db_path = tmp_path / "partitioned_loader.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([User(**row) for row in rows])
            session.commit()
    finally:
        engine.dispose()

    return SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )


def _create_event_db(tmp_path, rows: list[dict[str, object]]) -> SqlDatabaseConfig:
    db_path = tmp_path / "partitioned_events.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([Event(**row) for row in rows])
            session.commit()
    finally:
        engine.dispose()

    return SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )


def test_partitioned_request_requires_partition_column_for_range():
    with pytest.raises(ValidationError, match="requires partition_column"):
        SqlPartitionedLoadRequest(
            statement=select(User),
            model=User,
            partition_strategy="range",
        )


def test_partitioned_request_caps_max_concurrent_fetches():
    with pytest.raises(ValidationError, match="less than or equal to"):
        SqlPartitionedLoadRequest(
            statement=select(User),
            model=User,
            max_concurrent_fetches=MAX_PARTITION_FETCH_CONCURRENCY + 1,
        )


def test_partitioned_loader_rejects_inmemory_sqlite():
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with pytest.raises(ValueError, match="in-memory DSNs are not supported"):
        SqlPartitionedLoader(config)


def test_partitioned_loader_returns_lazy_dask_frame(tmp_path):
    config = _create_user_db(
        tmp_path,
        [
            {"id": 1, "status": "active", "description": "a"},
            {"id": 2, "status": "active", "description": "b"},
            {"id": 3, "status": "inactive", "description": "c"},
            {"id": 4, "status": "active", "description": "d"},
            {"id": 5, "status": "inactive", "description": "e"},
        ],
    )

    with SqlPartitionedLoader(config) as loader:
        plan = loader.plan(statement=select(User), model=User, chunk_size=2)
        frame = loader.load(statement=select(User), model=User, chunk_size=2)

    assert isinstance(frame, dd.DataFrame)
    assert plan.total_rows == 5
    assert len(plan.partitions) == 3
    assert frame.compute()["id"].tolist() == [1, 2, 3, 4, 5]


def test_partitioned_loader_uses_filtered_bounds_for_range_planning(tmp_path):
    config = _create_user_db(
        tmp_path,
        [
            {"id": 1, "status": "inactive", "description": "old"},
            {"id": 2, "status": "inactive", "description": "older"},
            {"id": 100, "status": "active", "description": "target-a"},
            {"id": 101, "status": "active", "description": "target-b"},
        ],
    )

    with SqlPartitionedLoader(config) as loader:
        plan = loader.plan(
            statement=select(User),
            model=User,
            filters={"status__exact": "active"},
            partition_strategy="range",
            partition_column="id",
            chunk_size=1,
        )
        frame = loader.load(
            statement=select(User),
            model=User,
            filters={"status__exact": "active"},
            partition_strategy="range",
            partition_column="id",
            chunk_size=1,
        )

    assert plan.total_rows == 2
    assert len(plan.partitions) == 2
    first_ints = [value for value in (plan.partitions[0].params or ()) if isinstance(value, int)]
    second_ints = [value for value in (plan.partitions[1].params or ()) if isinstance(value, int)]
    assert first_ints == [100, 101]
    assert second_ints == [101, 102]
    assert frame.compute()["id"].tolist() == [100, 101]


def test_partitioned_loader_propagates_partition_failures(tmp_path, monkeypatch):
    config = _create_user_db(
        tmp_path,
        [
            {"id": 1, "status": "active", "description": "a"},
            {"id": 2, "status": "active", "description": "b"},
        ],
    )

    def _failing_fetch(**_kwargs):
        raise RuntimeError("partition failure")

    monkeypatch.setattr(
        SqlPartitionedLoader,
        "_fetch_partition_static",
        staticmethod(_failing_fetch),
    )

    with SqlPartitionedLoader(config) as loader:
        frame = loader.load(statement=select(User), model=User, chunk_size=1)

    with pytest.raises(RuntimeError, match="partition failure"):
        frame.compute()


def test_partitioned_loader_coerces_date_columns_to_utc_timestamps(tmp_path):
    config = _create_event_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2046, 5, 31)},
            {"id": 2, "event_date": dt.date(2046, 6, 1)},
        ],
    )

    with SqlPartitionedLoader(config) as loader:
        frame = loader.load(statement=select(Event), model=Event, chunk_size=1)

    result = frame.compute().sort_values("id").reset_index(drop=True)

    assert result["id"].tolist() == [1, 2]
    assert str(result["event_date"].dtype) == "datetime64[ns, UTC]"
    assert result["event_date"].tolist() == [
        pd.Timestamp("2046-05-31T00:00:00Z"),
        pd.Timestamp("2046-06-01T00:00:00Z"),
    ]


def test_arrow_partition_alignment_regression_for_date_strings():
    result = SqlPartitionExecutor._arrow_align_and_coerce_partition(
        rows=[("2046-05-31",), ("2046-06-01",)],
        columns=["event_date"],
        meta_dtypes={"event_date": "datetime64[ns, UTC]"},
    )

    assert str(result["event_date"].dtype) == "datetime64[ns, UTC]"
    assert result["event_date"].tolist() == [
        pd.Timestamp("2046-05-31T00:00:00Z"),
        pd.Timestamp("2046-06-01T00:00:00Z"),
    ]


def test_partitioned_loader_computes_with_distributed_client(tmp_path):
    distributed = pytest.importorskip("dask.distributed")
    Client = distributed.Client
    LocalCluster = distributed.LocalCluster

    config = _create_user_db(
        tmp_path,
        [
            {"id": 1, "status": "active", "description": "a"},
            {"id": 2, "status": "active", "description": "b"},
            {"id": 3, "status": "inactive", "description": "c"},
            {"id": 4, "status": "active", "description": "d"},
        ],
    )

    with (
        LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=False,
            dashboard_address=":0",
        ) as cluster,
        Client(cluster),
    ):
        with SqlPartitionedLoader(config) as loader:
            frame = loader.load(statement=select(User), model=User, chunk_size=2)

        assert isinstance(frame, dd.DataFrame)
        assert frame.compute().sort_values("id")["id"].tolist() == [1, 2, 3, 4]


def test_partition_executor_streams_partition_rows_without_fetchall():
    class FakeResult:
        def __init__(self) -> None:
            self._batches = [
                [(1, "active"), (2, "inactive")],
                [(3, "active")],
            ]

        def keys(self):
            return ["id", "status"]

        def fetchmany(self, _size: int):
            if self._batches:
                return self._batches.pop(0)
            return []

        def fetchall(self):
            raise AssertionError("fetchall() should not be used")

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def exec_driver_sql(self, _sql, _params=None):
            return FakeResult()

    class FakeEngine:
        def connect(self):
            return FakeConnection()

    meta_dtypes = {"id": "Int64", "status": "string"}
    original = SqlPartitionExecutor.fetch_partition.__globals__["_get_cached_worker_sync_engine"]
    SqlPartitionExecutor.fetch_partition.__globals__["_get_cached_worker_sync_engine"] = (
        lambda _config: FakeEngine()
    )
    try:
        result = SqlPartitionExecutor.fetch_partition(
            config=None,
            gate_key="stream-test",
            max_concurrent_fetches=1,
            partition=type("Partition", (), {"sql": "SELECT 1", "params": {}})(),
            meta_dtypes=meta_dtypes,
            use_arrow=True,
        )
    finally:
        SqlPartitionExecutor.fetch_partition.__globals__["_get_cached_worker_sync_engine"] = (
            original
        )

    assert result["id"].tolist() == [1, 2, 3]
    assert result["status"].tolist() == ["active", "inactive", "active"]


def test_align_and_coerce_partition_skips_redundant_validation(monkeypatch):
    monkeypatch.setattr(
        "boti_data.db.partitioned_execution.validate_schema",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("validate_schema should not run")
        ),
        raising=False,
    )
    frame = pd.DataFrame({"id": ["1"], "status": ["active"]})

    result = SqlPartitionExecutor.align_and_coerce_partition(
        frame,
        {"id": "Int64", "status": "string"},
    )

    assert result["id"].tolist() == [1]
    assert result["status"].tolist() == ["active"]


def test_estimated_rows_skips_count_query(tmp_path, monkeypatch):
    config = _create_user_db(
        tmp_path,
        [{"id": i, "status": "active", "description": str(i)} for i in range(100)],
    )

    count_calls = 0
    original_count = SqlPartitionPlanner.count_rows
    def _counting(*args: Any, **_kwargs: Any) -> int:
        nonlocal count_calls
        count_calls += 1
        return original_count(*args, **_kwargs)

    monkeypatch.setattr(SqlPartitionPlanner, "count_rows", _counting)

    request = SqlPartitionedLoadRequest(
        statement=select(User),
        model=User,
        chunk_size=50_000,
        estimated_rows=100,
    )

    with SqlPartitionedLoader(config) as loader:
        plan = loader.plan(request=request)
        frame = loader.load(request=request)

    assert count_calls == 0
    assert len(plan.partitions) == 1
    assert plan.total_rows == 100
    assert frame.compute()["id"].tolist() == list(range(100))


def test_estimated_rows_zero_returns_empty_plan(tmp_path, monkeypatch):
    config = _create_user_db(tmp_path, [{"id": 1, "status": "a", "description": "b"}])

    count_calls = 0
    original_count = SqlPartitionPlanner.count_rows
    def _counting(*args: Any, **_kwargs: Any) -> int:
        nonlocal count_calls
        count_calls += 1
        return original_count(*args, **_kwargs)

    monkeypatch.setattr(SqlPartitionPlanner, "count_rows", _counting)

    request = SqlPartitionedLoadRequest(
        statement=select(User),
        model=User,
        chunk_size=50_000,
        estimated_rows=0,
    )

    with SqlPartitionedLoader(config) as loader:
        plan = loader.plan(request=request)

    assert count_calls == 0
    assert len(plan.partitions) == 0
    assert plan.total_rows == 0


def test_estimated_rows_greater_than_chunk_still_counts(tmp_path, monkeypatch):
    config = _create_user_db(
        tmp_path,
        [{"id": i, "status": "active", "description": str(i)} for i in range(100)],
    )

    count_calls = 0
    original_count = SqlPartitionPlanner.count_rows
    def _counting(*args: Any, **_kwargs: Any) -> int:
        nonlocal count_calls
        count_calls += 1
        return original_count(*args, **_kwargs)

    monkeypatch.setattr(SqlPartitionPlanner, "count_rows", _counting)

    request = SqlPartitionedLoadRequest(
        statement=select(User),
        model=User,
        chunk_size=10,
        estimated_rows=100,
    )

    with SqlPartitionedLoader(config) as loader:
        plan = loader.plan(request=request)

    assert count_calls >= 1
    assert plan.total_rows == 100
    assert len(plan.partitions) == 10


def test_partitioned_loader_accepts_prevalidated_request(tmp_path, monkeypatch):
    config = _create_user_db(
        tmp_path,
        [
            {"id": 1, "status": "active", "description": "a"},
            {"id": 2, "status": "inactive", "description": "b"},
        ],
    )
    request = SqlPartitionedLoadRequest(
        statement=select(User),
        model=User,
        chunk_size=1,
    )
    monkeypatch.setattr(
        SqlPartitionedLoadRequest,
        "model_validate",
        classmethod(
            lambda cls, *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("model_validate should not run")
            )
        ),
    )

    with SqlPartitionedLoader(config) as loader:
        frame = loader.load(request=request)

    assert frame.compute()["id"].tolist() == [1, 2]
