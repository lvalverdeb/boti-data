"""R5: resumable partition plans; an interrupted load fetches only what is missing."""

from __future__ import annotations

import dask
import pandas as pd
import pytest
from sqlalchemy import select

from boti_data.db import (
    FileCheckpointStore,
    PlanCheckpoint,
    SqlPartitionedLoader,
    SqlPartitionedLoadRequest,
)
from boti_data.db.partitioned_checkpoint import CheckpointedFetch
from boti_data.db.partitioned_execution import SqlPartitionExecutor
from boti_data.db.partitioned_types import SqlPartitionPlan, SqlPartitionSpec

from ._partitioned_sql_loader_shared import User, _create_user_db

ROWS = 1_000
CHUNK = 10  # 100 partitions


class _Interrupt(RuntimeError):
    pass


class _CountingFetch:
    """Counts real database fetches; raises once ``fail_after`` partitions have been fetched."""

    def __init__(self, fail_after: int | None) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def __call__(self, **kwargs):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise _Interrupt("simulated crash")
        return SqlPartitionExecutor.fetch_partition(**kwargs)


def _request(**extra) -> SqlPartitionedLoadRequest:
    return SqlPartitionedLoadRequest(
        statement=select(User),
        model=User,
        chunk_size=CHUNK,
        as_pandas=True,
        max_concurrent_fetches=1,
        **extra,
    )


def test_plan_json_round_trip() -> None:
    plan = SqlPartitionPlan(
        total_rows=3,
        strategy="offset",
        partitions=(
            SqlPartitionSpec("select 1", (1, 2)),
            SqlPartitionSpec("select 2", {"a": 1}),
            SqlPartitionSpec("select 3"),
        ),
        meta_dtypes={"id": "Int64"},
    )
    assert SqlPartitionPlan.from_json(plan.to_json()) == plan


def test_interrupted_load_resumes_fetching_only_the_missing_partitions(
    tmp_path, monkeypatch
) -> None:
    config = _create_user_db(
        tmp_path, [{"id": i, "status": "a", "description": f"d{i}"} for i in range(1, ROWS + 1)]
    )
    store = FileCheckpointStore(tmp_path / "checkpoints")
    failed_after = 37

    # The threaded scheduler keeps in-flight partitions running after the first
    # error, which would make "exactly N fetched" racy; run tasks one at a time.
    with SqlPartitionedLoader(config) as loader, dask.config.set(scheduler="synchronous"):
        counting = _CountingFetch(fail_after=failed_after)
        monkeypatch.setattr(loader, "_fetch_partition_static", counting)
        with pytest.raises(_Interrupt):
            loader.load_request(_request(), checkpoint=PlanCheckpoint(store, "users"))
        assert counting.calls == failed_after + 1

    stored = PlanCheckpoint.load(store, "users")
    assert stored is not None and len(stored.plan.partitions) == 100
    assert len(stored.pending) == 100 - failed_after

    with SqlPartitionedLoader(config) as loader:
        counting = _CountingFetch(fail_after=None)
        monkeypatch.setattr(loader, "_fetch_partition_static", counting)
        frame = loader.load_request(
            _request(resume=True), checkpoint=PlanCheckpoint(store, "users")
        )
        assert counting.calls == 100 - failed_after

    assert len(frame) == ROWS
    assert sorted(frame["id"].tolist()) == list(range(1, ROWS + 1))
    assert PlanCheckpoint.load(store, "users").complete


def test_without_resume_the_checkpoint_starts_afresh(tmp_path, monkeypatch) -> None:
    config = _create_user_db(
        tmp_path, [{"id": i, "status": "a", "description": "d"} for i in range(1, 51)]
    )
    store = FileCheckpointStore(tmp_path / "cp")
    with SqlPartitionedLoader(config) as loader:
        loader.load_request(_request(), checkpoint=PlanCheckpoint(store, "k"))
        counting = _CountingFetch(fail_after=None)
        monkeypatch.setattr(loader, "_fetch_partition_static", counting)
        loader.load_request(_request(), checkpoint=PlanCheckpoint(store, "k"))
        assert counting.calls == 5
        counting.calls = 0
        loader.load_request(_request(resume=True), checkpoint=PlanCheckpoint(store, "k"))
        assert counting.calls == 0


def test_checkpointed_fetch_survives_pickling(tmp_path) -> None:
    import pickle

    store = FileCheckpointStore(tmp_path / "cp")
    checkpoint = PlanCheckpoint(store, "p")
    checkpoint.begin(
        SqlPartitionPlan(1, "offset", (SqlPartitionSpec("select 1"),), {"id": "Int64"})
    )
    fetch = CheckpointedFetch(checkpoint, SqlPartitionExecutor.fetch_partition)
    restored = pickle.loads(pickle.dumps(fetch))
    assert restored.checkpoint.key == "p" and restored.checkpoint.pending
    checkpoint.mark_done(
        SqlPartitionSpec("select 1"), pd.DataFrame({"id": pd.array([1], dtype="Int64")})
    )
    assert checkpoint.stored_frame(SqlPartitionSpec("select 1"))["id"].tolist() == [1]
