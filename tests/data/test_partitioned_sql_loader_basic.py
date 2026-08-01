"""
Tests for the partition-aware SQL -> Dask loader: request validation and
end-to-end plan/load correctness.

Split out of test_partitioned_sql_loader.py purely for god-module/long-file
headroom.
"""

from __future__ import annotations

import datetime as dt
import gc

import dask.dataframe as dd
import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import select

from boti_data.db import (
    SqlDatabaseConfig,
    SqlDatabaseResource,
    SqlPartitionedLoader,
    SqlPartitionedLoadRequest,
)
from boti_data.db.partitioned_types import MAX_PARTITION_FETCH_CONCURRENCY
from boti_data.gateway.loaders import load_sql_partitioned

from ._partitioned_sql_loader_shared import Event, User, _create_event_db, _create_user_db


def test_partitioned_request_requires_partition_column_for_range() -> None:
    with pytest.raises(ValidationError, match="requires partition_column"):
        SqlPartitionedLoadRequest(
            statement=select(User),
            model=User,
            partition_strategy="range",
        )


def test_partitioned_request_caps_max_concurrent_fetches() -> None:
    with pytest.raises(ValidationError, match="less than or equal to"):
        SqlPartitionedLoadRequest(
            statement=select(User),
            model=User,
            max_concurrent_fetches=MAX_PARTITION_FETCH_CONCURRENCY + 1,
        )


def test_partitioned_loader_rejects_inmemory_sqlite() -> None:
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with pytest.raises(ValueError, match="in-memory DSNs are not supported"):
        SqlPartitionedLoader(config)


def test_partitioned_loader_rejects_inmemory_duckdb() -> None:
    config = SqlDatabaseConfig(
        connection_url="duckdb:///:memory:",
        query_only=False,
    )

    with pytest.raises(ValueError, match="in-memory DSNs are not supported"):
        SqlPartitionedLoader(config)


def test_partitioned_loader_rejects_writable_duckdb(tmp_path) -> None:
    """DuckDB opening a file in its default (read-write) mode claims an
    exclusive lock on connect, even for a connection that never writes — so a
    second worker process would block on the very first one to connect,
    regardless of query_only. Only query_only=True (native read-only mode) is
    safe for distributed workers."""
    config = SqlDatabaseConfig(
        connection_url=f"duckdb:///{tmp_path / 'writable.duckdb'}",
        query_only=False,
    )

    with pytest.raises(ValueError, match="requires query_only=True"):
        SqlPartitionedLoader(config)


def test_partitioned_loader_accepts_readonly_duckdb(tmp_path) -> None:
    config = SqlDatabaseConfig(
        connection_url=f"duckdb:///{tmp_path / 'readonly.duckdb'}",
        query_only=True,
    )

    with SqlPartitionedLoader(config):
        pass


def test_partitioned_loader_returns_lazy_dask_frame(tmp_path) -> None:
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


def test_load_sql_partitioned_closes_loader_without_leak_warning(tmp_path) -> None:
    """The internal SqlPartitionedLoader built by load_sql_partitioned() must be
    closed deterministically rather than relying on GC -- previously it was
    constructed and never closed, logging "garbage collected without being
    closed" under load. resource is caller-owned here, so closing the loader
    must not close resource out from under the caller."""
    config = _create_user_db(tmp_path, [{"id": 1, "status": "active", "description": "a"}])
    resource = SqlDatabaseResource(config)
    warnings_seen: list[str] = []
    resource.logger.warning = lambda msg, *a, **kw: warnings_seen.append(msg)
    try:
        request = SqlPartitionedLoadRequest.model_validate(
            {"statement": select(User), "model": User}
        )
        frame = load_sql_partitioned(config, resource, request)
        gc.collect()

        assert not any("garbage collected" in msg for msg in warnings_seen)
        assert frame.compute()["id"].tolist() == [1]
        assert not resource.closed
    finally:
        resource.close()


def test_partitioned_loader_uses_filtered_bounds_for_range_planning(tmp_path) -> None:
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


def test_partitioned_loader_propagates_partition_failures(tmp_path, monkeypatch) -> None:
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


def test_partitioned_loader_coerces_date_columns_to_utc_timestamps(tmp_path) -> None:
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


def test_partitioned_loader_computes_with_distributed_client(tmp_path) -> None:
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


def test_partitioned_loader_accepts_prevalidated_request(tmp_path, monkeypatch) -> None:
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
