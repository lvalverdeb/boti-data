"""The partition-fetch gate is keyed on the database only and bounds the cluster."""

from __future__ import annotations

import threading
import time

import pytest

from boti_data.db.partitioned_fetch_gate import _fetch_gate_for, _gate_name, _timed_fetch_gate

_state = {"active": 0, "max_seen": 0, "runs": 0}
_state_lock = threading.Lock()


def test_gate_name_carries_nothing_of_the_dsn() -> None:
    name = _gate_name("postgresql://alice:s3cret@db.internal/orders|query_only=True")
    assert "s3cret" not in name and "alice" not in name and "db.internal" not in name
    assert name == _gate_name("postgresql://alice:s3cret@db.internal/orders|query_only=True")


def test_gate_is_keyed_on_the_database_not_the_limit() -> None:
    assert _fetch_gate_for("same-dsn", 4) is _fetch_gate_for("same-dsn", 8)
    assert _fetch_gate_for("same-dsn", 4) is not _fetch_gate_for("other-dsn", 4)


def _gated_fetch(limit: int) -> int:
    with _timed_fetch_gate("cluster-dsn", limit):
        with _state_lock:
            _state["active"] += 1
            _state["max_seen"] = max(_state["max_seen"], _state["active"])
        time.sleep(0.15)
        with _state_lock:
            _state["active"] -= 1
            _state["runs"] += 1
    return 1


def test_gate_bounds_fetches_across_the_whole_cluster() -> None:
    """8 worker threads on 4 workers, limit 2: never more than 2 fetches at once."""
    pytest.importorskip("dask.distributed")
    from boti_dask import dask_session

    _state.update(active=0, max_seen=0, runs=0)
    with dask_session(
        cluster_kwargs={
            "n_workers": 4,
            "threads_per_worker": 2,
            "processes": False,
            "dashboard_address": ":0",
        }
    ) as client:
        futures = [client.submit(_gated_fetch, 2, pure=False) for _ in range(8)]
        assert sum(client.gather(futures)) == 8
    assert _state["runs"] == 8
    assert _state["max_seen"] == 2


def _gate_type_name_inside_task() -> str:
    return type(_fetch_gate_for("type-probe-dsn", 2)).__name__


def test_gate_inside_a_task_is_the_scheduler_semaphore() -> None:
    """On a cluster the gate must be distributed's Semaphore, not a process lock."""
    pytest.importorskip("dask.distributed")
    boti_dask = pytest.importorskip("boti_dask")
    if not hasattr(boti_dask, "cluster_semaphore"):
        pytest.skip("boti-dask < 1.1.0 has no cluster_semaphore; gate is per process")
    from boti_dask import dask_session

    with dask_session(
        cluster_kwargs={
            "n_workers": 1,
            "threads_per_worker": 1,
            "processes": False,
            "dashboard_address": ":0",
        }
    ) as client:
        assert (
            client.submit(_gate_type_name_inside_task, pure=False).result(timeout=20) == "Semaphore"
        )
