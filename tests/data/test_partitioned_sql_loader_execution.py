"""
Tests for the partition-aware SQL -> Dask loader: executor internals
(row streaming, schema coercion) and estimated-row-count planning shortcuts.

Split out of test_partitioned_sql_loader.py purely for god-module/long-file
headroom.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import select

from boti_data.db import SqlPartitionedLoader, SqlPartitionedLoadRequest
from boti_data.db.partitioned_execution import SqlPartitionExecutor
from boti_data.db.partitioned_planner import SqlPartitionPlanner

from ._partitioned_sql_loader_shared import User, _create_user_db


class _CallCounter:
    """Mutable call-count holder for _counting_wrapper()'s closure."""

    def __init__(self) -> None:
        self.calls = 0


def _counting_wrapper(original: Any, counter: _CallCounter) -> Any:
    """Wrap *original* (an unbound method) to tally calls in *counter*.

    Returned as a plain function rather than a callable instance: it gets
    monkeypatched onto ``SqlPartitionPlanner`` as a class attribute, and only
    a plain function auto-binds ``self`` via the descriptor protocol the way
    the real method would.
    """

    def _counting(*args: Any, **kwargs: Any) -> Any:
        counter.calls += 1
        return original(*args, **kwargs)

    return _counting


class _StreamingFakeResult:
    def __init__(self) -> None:
        self._batches = [
            [(1, "active"), (2, "inactive")],
            [(3, "active")],
        ]

    def keys(self) -> list[str]:
        return ["id", "status"]

    def fetchmany(self, _size: int) -> list[tuple[int, str]]:
        if self._batches:
            return self._batches.pop(0)
        return []

    def fetchall(self) -> None:
        raise AssertionError("fetchall() should not be used")


class _StreamingFakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def exec_driver_sql(self, _sql, _params=None) -> _StreamingFakeResult:
        return _StreamingFakeResult()


class _StreamingFakeEngine:
    def connect(self) -> _StreamingFakeConnection:
        return _StreamingFakeConnection()


def test_arrow_partition_alignment_regression_for_date_strings() -> None:
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


def test_partition_executor_streams_partition_rows_without_fetchall() -> None:
    meta_dtypes = {"id": "Int64", "status": "string"}
    original = SqlPartitionExecutor.fetch_partition.__globals__["_get_cached_worker_sync_engine"]
    SqlPartitionExecutor.fetch_partition.__globals__["_get_cached_worker_sync_engine"] = (
        lambda _config: _StreamingFakeEngine()
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


def test_align_and_coerce_partition_skips_redundant_validation(monkeypatch) -> None:
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


def test_estimated_rows_skips_count_query(tmp_path, monkeypatch) -> None:
    config = _create_user_db(
        tmp_path,
        [{"id": i, "status": "active", "description": str(i)} for i in range(100)],
    )

    counter = _CallCounter()
    monkeypatch.setattr(
        SqlPartitionPlanner,
        "count_rows",
        _counting_wrapper(SqlPartitionPlanner.count_rows, counter),
    )

    request = SqlPartitionedLoadRequest(
        statement=select(User),
        model=User,
        chunk_size=50_000,
        estimated_rows=100,
    )

    with SqlPartitionedLoader(config) as loader:
        plan = loader.plan(request=request)
        frame = loader.load(request=request)

    assert counter.calls == 0
    assert len(plan.partitions) == 1
    assert plan.total_rows == 100
    assert frame.compute()["id"].tolist() == list(range(100))


def test_estimated_rows_zero_returns_empty_plan(tmp_path, monkeypatch) -> None:
    config = _create_user_db(tmp_path, [{"id": 1, "status": "a", "description": "b"}])

    counter = _CallCounter()
    monkeypatch.setattr(
        SqlPartitionPlanner,
        "count_rows",
        _counting_wrapper(SqlPartitionPlanner.count_rows, counter),
    )

    request = SqlPartitionedLoadRequest(
        statement=select(User),
        model=User,
        chunk_size=50_000,
        estimated_rows=0,
    )

    with SqlPartitionedLoader(config) as loader:
        plan = loader.plan(request=request)

    assert counter.calls == 0
    assert len(plan.partitions) == 0
    assert plan.total_rows == 0


def test_estimated_rows_greater_than_chunk_still_counts(tmp_path, monkeypatch) -> None:
    config = _create_user_db(
        tmp_path,
        [{"id": i, "status": "active", "description": str(i)} for i in range(100)],
    )

    counter = _CallCounter()
    monkeypatch.setattr(
        SqlPartitionPlanner,
        "count_rows",
        _counting_wrapper(SqlPartitionPlanner.count_rows, counter),
    )

    request = SqlPartitionedLoadRequest(
        statement=select(User),
        model=User,
        chunk_size=10,
        estimated_rows=100,
    )

    with SqlPartitionedLoader(config) as loader:
        plan = loader.plan(request=request)

    assert counter.calls >= 1
    assert plan.total_rows == 100
    assert len(plan.partitions) == 10
