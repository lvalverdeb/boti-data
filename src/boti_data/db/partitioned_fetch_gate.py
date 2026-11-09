"""Partition-fetch concurrency gate and its contention accounting.

Split out of partitioned_execution.py purely for long-file headroom. Every name
here is re-exported from partitioned_execution.py so existing import paths keep
working unchanged.

The gate bounds how many partition fetches run at once, and the counters record
how long fetches spent queueing for a slot. That second half exists because the
gate is otherwise invisible: the ``diagnostics=True`` stage timers (``db_fetch``,
``prepare_stmt``, ``load_plan``) are all measured *inside* the gate, so under
saturation total elapsed time rises while every stage timer still looks healthy.
An idle database plus timing-out callers is the classic symptom, and without
these counters it is indistinguishable from a database that is simply slow.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from boti_data.telemetry import current_provider

_module_log = logging.getLogger(__name__)

# The gate is keyed on the database identity alone, never on the limit: the
# resource being protected is "that database", so two loads with different
# limits must share one gate (the first limit seen wins) rather than add up.
#
# On a Dask cluster the gate is boti-dask's cluster_semaphore, held on the
# scheduler and shared by every worker, so max_concurrent_fetches bounds the
# whole cluster. With no running client (local threads, tests) the same call
# returns a process-wide BoundedSemaphore, as before.

# 16 hex characters of SHA-256 (64 bits) is plenty to keep distinct databases
# apart while keeping the scheduler-side name short.
_GATE_NAME_DIGEST_CHARS = 16

_LOCAL_GATES: dict[str, threading.BoundedSemaphore] = {}
_LOCAL_GATES_LOCK = threading.Lock()


def _gate_name(gate_key: str) -> str:
    """A stable name for the gate that carries nothing of the DSN (it travels to
    the scheduler and shows in dashboards; the key may embed credentials)."""
    digest = hashlib.sha256(gate_key.encode()).hexdigest()[:_GATE_NAME_DIGEST_CHARS]
    return f"boti-data:fetch-gate:{digest}"


def _local_fetch_gate(name: str, max_concurrent_fetches: int) -> threading.BoundedSemaphore:
    with _LOCAL_GATES_LOCK:
        gate = _LOCAL_GATES.get(name)
        if gate is None:
            gate = threading.BoundedSemaphore(max_concurrent_fetches)
            _LOCAL_GATES[name] = gate
        return gate


def _fetch_gate_for(gate_key: str, max_concurrent_fetches: int) -> Any:
    name = _gate_name(gate_key)
    try:
        from boti_dask import cluster_semaphore
    except ImportError:  # boti-dask < 1.1.0: process-wide only
        _module_log.debug("boti-dask without cluster_semaphore; fetch gate is per process")
        return _local_fetch_gate(name, max_concurrent_fetches)
    return cluster_semaphore(name, max_concurrent_fetches)


@dataclass(frozen=True)
class GateWaitStats:
    """Time spent waiting for a partition-fetch slot, accumulated per process.

    ``max_wait_seconds`` is cumulative for the life of the process and cannot be
    differenced into a per-load figure the way the other three can (a running
    maximum carries no record of when it was set). :meth:`since` therefore
    carries it through unchanged; it reads as "worst wait seen in this process
    so far", not "worst wait during this load".
    """

    fetches: int = 0
    waited_fetches: int = 0
    total_wait_seconds: float = 0.0
    max_wait_seconds: float = 0.0

    def since(self, baseline: GateWaitStats | None) -> GateWaitStats:
        """Return the stats accumulated since ``baseline`` was captured."""
        if baseline is None:
            return self
        return GateWaitStats(
            fetches=self.fetches - baseline.fetches,
            waited_fetches=self.waited_fetches - baseline.waited_fetches,
            total_wait_seconds=self.total_wait_seconds - baseline.total_wait_seconds,
            max_wait_seconds=self.max_wait_seconds,
        )


# An uncontended BoundedSemaphore.acquire() returns in ~1us; anything at or above
# a millisecond means the fetch genuinely queued behind another one.
_GATE_WAIT_EPSILON_SECONDS = 0.001

_GATE_WAIT_STATS: dict[str, GateWaitStats] = {}
_GATE_WAIT_LOCK = threading.Lock()


def _record_gate_wait(gate_key: str, waited_seconds: float) -> None:
    current_provider().histogram("boti_data.gate.wait_seconds").record(
        waited_seconds, gate=_gate_name(gate_key)
    )
    with _GATE_WAIT_LOCK:
        current = _GATE_WAIT_STATS.get(gate_key, GateWaitStats())
        _GATE_WAIT_STATS[gate_key] = GateWaitStats(
            fetches=current.fetches + 1,
            waited_fetches=current.waited_fetches
            + (1 if waited_seconds >= _GATE_WAIT_EPSILON_SECONDS else 0),
            total_wait_seconds=current.total_wait_seconds + waited_seconds,
            max_wait_seconds=max(current.max_wait_seconds, waited_seconds),
        )


def fetch_gate_stats() -> dict[str, GateWaitStats]:
    """Snapshot partition-fetch gate contention for **this process**, by gate key.

    Scope matters for interpreting the numbers. The gate itself is cluster-wide
    on a Dask cluster, but the wait counters are not: partition fetches are
    dispatched as Dask tasks, so in a distributed run each worker process
    accumulates its own counters and this function only ever sees the calling
    process's share.
    A lazy load returns before any fetch has run at all, so call this after
    ``.compute()``, not after the load call returns.
    """
    with _GATE_WAIT_LOCK:
        return dict(_GATE_WAIT_STATS)


def reset_fetch_gate_stats() -> None:
    """Clear accumulated gate-wait counters for this process."""
    with _GATE_WAIT_LOCK:
        _GATE_WAIT_STATS.clear()


def format_gate_wait_suffix(gate_key: str, baseline: GateWaitStats | None) -> str:
    """Format gate contention for a diagnostics line, or ``""`` if nothing ran here.

    Shared by the sync loader and the async gateway so both diagnostics lines
    report contention identically.

    Returning empty is deliberate rather than reporting zeros. A lazy load
    returns a Dask graph before any partition has been fetched, and in a
    distributed run the fetches then happen inside other worker processes
    entirely — ``gate_wait_total=0.000s`` there would read as "no contention"
    when the truth is "not measured here". Callers wanting figures for a lazy
    load should read :func:`fetch_gate_stats` after ``.compute()``.
    """
    delta = fetch_gate_stats().get(gate_key, GateWaitStats()).since(baseline)
    if delta.fetches <= 0:
        return ""
    return (
        f" gate_waits={delta.waited_fetches}/{delta.fetches}"
        f" gate_wait_total={delta.total_wait_seconds:.3f}s"
        f" gate_wait_max_seen={delta.max_wait_seconds:.3f}s"
    )


@contextmanager
def _timed_fetch_gate(gate_key: str, max_concurrent_fetches: int) -> Iterator[None]:
    """Hold a fetch slot, recording how long it took to get one."""
    gate = _fetch_gate_for(gate_key, max_concurrent_fetches)
    started = perf_counter()
    gate.acquire()
    _record_gate_wait(gate_key, perf_counter() - started)
    try:
        yield
    finally:
        gate.release()
