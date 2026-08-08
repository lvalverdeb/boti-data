"""
Tests for partition-fetch gate contention accounting.

The gate timer exists to make one specific failure mode visible: the database
is nearly idle while requests queue behind a saturated fetch gate. These tests
therefore assert that real contention is *measured*, not merely that counters
increment.
"""

from __future__ import annotations

import threading
from time import perf_counter

import pandas as pd
import pytest

from boti_data.db.partitioned_execution import (
    GateWaitStats,
    _timed_fetch_gate,
    fetch_gate_stats,
    reset_fetch_gate_stats,
)
from boti_data.gateway.sql_partitioned_exec import _log_completion

from ._partitioned_sql_loader_shared import User, _create_user_db


@pytest.fixture(autouse=True)
def _clean_gate_stats():
    reset_fetch_gate_stats()
    yield
    reset_fetch_gate_stats()


def test_uncontended_fetch_records_no_wait() -> None:
    with _timed_fetch_gate("gate-uncontended", 4):
        pass

    stats = fetch_gate_stats()["gate-uncontended"]
    assert stats.fetches == 1
    # An uncontended acquire returns in ~1us, well under the 1ms epsilon.
    assert stats.waited_fetches == 0
    assert stats.max_wait_seconds < 0.001


def test_saturated_gate_records_the_wait() -> None:
    """A fetch blocked behind a full gate must show up as measured wait time."""
    gate_key = "gate-saturated"
    holder_acquired = threading.Event()
    release_holder = threading.Event()

    def _hold_the_only_slot() -> None:
        with _timed_fetch_gate(gate_key, 1):
            holder_acquired.set()
            release_holder.wait(timeout=5)

    holder = threading.Thread(target=_hold_the_only_slot)
    holder.start()
    assert holder_acquired.wait(timeout=5), "holder never acquired the slot"

    # The gate has one slot and it is taken, so this waits until released.
    def _delayed_release() -> None:
        release_holder.set()

    timer = threading.Timer(0.05, _delayed_release)
    timer.start()
    started = perf_counter()
    with _timed_fetch_gate(gate_key, 1):
        blocked_for = perf_counter() - started
    holder.join(timeout=5)
    timer.cancel()

    assert blocked_for >= 0.05, "test did not actually block"
    stats = fetch_gate_stats()[gate_key]
    assert stats.fetches == 2
    assert stats.waited_fetches == 1, "the blocked fetch was not counted as waiting"
    assert stats.max_wait_seconds >= 0.05
    assert stats.total_wait_seconds >= 0.05


def test_gate_is_released_when_the_body_raises() -> None:
    """A failing fetch must not leak its slot, or the gate deadlocks."""
    with pytest.raises(RuntimeError):
        with _timed_fetch_gate("gate-raises", 1):
            raise RuntimeError("fetch blew up")

    # If the slot leaked, this second acquire would block forever.
    completed = threading.Event()

    def _second_acquire() -> None:
        with _timed_fetch_gate("gate-raises", 1):
            completed.set()

    thread = threading.Thread(target=_second_acquire)
    thread.start()
    thread.join(timeout=5)
    assert completed.is_set(), "gate slot leaked after an exception"


def test_stats_are_isolated_per_gate_key() -> None:
    with _timed_fetch_gate("gate-a", 2):
        pass
    with _timed_fetch_gate("gate-b", 2):
        pass
    with _timed_fetch_gate("gate-b", 2):
        pass

    stats = fetch_gate_stats()
    assert stats["gate-a"].fetches == 1
    assert stats["gate-b"].fetches == 2


def test_snapshot_does_not_alias_live_state() -> None:
    with _timed_fetch_gate("gate-snapshot", 2):
        pass
    snapshot = fetch_gate_stats()
    with _timed_fetch_gate("gate-snapshot", 2):
        pass

    assert snapshot["gate-snapshot"].fetches == 1, "snapshot mutated after capture"
    assert fetch_gate_stats()["gate-snapshot"].fetches == 2


class TestGateWaitStatsSince:
    def test_subtracts_additive_counters(self) -> None:
        baseline = GateWaitStats(
            fetches=3, waited_fetches=1, total_wait_seconds=0.5, max_wait_seconds=0.4
        )
        current = GateWaitStats(
            fetches=10, waited_fetches=4, total_wait_seconds=2.0, max_wait_seconds=0.9
        )
        delta = current.since(baseline)

        assert delta.fetches == 7
        assert delta.waited_fetches == 3
        assert delta.total_wait_seconds == pytest.approx(1.5)

    def test_carries_max_through_because_it_is_not_differenceable(self) -> None:
        """A running maximum cannot be differenced; it stays cumulative by design."""
        baseline = GateWaitStats(max_wait_seconds=0.4)
        current = GateWaitStats(fetches=1, max_wait_seconds=0.9)

        assert current.since(baseline).max_wait_seconds == 0.9

    def test_no_baseline_returns_the_reading_unchanged(self) -> None:
        current = GateWaitStats(fetches=2, waited_fetches=1, total_wait_seconds=0.3)
        assert current.since(None) == current


class TestLoaderDiagnosticsSuffix:
    """The suffix must appear only when fetches actually ran in this process."""

    @staticmethod
    def _rows(count: int) -> list[dict[str, object]]:
        return [
            {"id": i, "status": f"status-{i % 3}", "description": f"user-{i}"}
            for i in range(1, count + 1)
        ]

    def _run(self, tmp_path, *, as_pandas: bool) -> tuple[str, object]:
        """Run a diagnostics load and return its completion line plus the result.

        boti's Logger writes through its own stdout handler rather than
        propagating to the root logger, so ``caplog`` never sees these records —
        recording ``logger.info`` directly is what actually observes them.
        """
        from sqlalchemy import select

        from boti_data.db import (
            SqlDatabaseResource,
            SqlPartitionedLoader,
            SqlPartitionedLoadRequest,
        )

        config = _create_user_db(tmp_path, self._rows(40))
        loader = SqlPartitionedLoader(config, resource=SqlDatabaseResource(config))
        recorded: list[str] = []
        original_info = loader.logger.info
        loader.logger.info = lambda message, *a, **kw: (  # type: ignore[method-assign]
            recorded.append(str(message)),
            original_info(message, *a, **kw),
        )[-1]
        try:
            result = loader.load_request(
                SqlPartitionedLoadRequest(
                    statement=select(User),
                    model=User,
                    chunk_size=10,
                    as_pandas=as_pandas,
                    diagnostics=True,
                )
            )
        finally:
            loader.logger.info = original_info  # type: ignore[method-assign]
            loader.close()

        completed = [line for line in recorded if "load completed" in line]
        assert completed, f"no completion diagnostics line was logged; got {recorded}"
        return completed[-1], result

    def test_eager_load_reports_gate_wait(self, tmp_path) -> None:
        line, _ = self._run(tmp_path, as_pandas=True)

        # 4 partitions at chunk_size=10 over 40 rows, all fetched in-process.
        assert "gate_waits=0/4" in line, line
        assert "gate_wait_total=" in line
        assert "gate_wait_max_seen=" in line

    def test_lazy_load_omits_gate_wait_rather_than_reporting_zero(self, tmp_path) -> None:
        """Fetches have not run yet, so a 0.000s reading would be a lie."""
        line, result = self._run(tmp_path, as_pandas=False)

        assert "gate_wait" not in line, (
            f"lazy load reported a gate wait before any partition was fetched: {line}"
        )
        assert result.npartitions >= 1


class TestGatewayDiagnosticsSuffix:
    """The async gateway logs its own completion line; it must report contention too.

    Regression guard: the gateway builds its own gate key and has a separate
    ``_log_completion``, so wiring only the sync loader would leave the
    DataHelper/facade path silently without the fields the README documents.
    """

    @staticmethod
    def _run(request_diagnostics: bool, gate_key: str, baseline=None) -> str:
        recorded: list[str] = []

        class _Logger:
            @staticmethod
            def info(message, *args, **kwargs) -> None:
                recorded.append(str(message))

        class _Resource:
            logger = _Logger()

        class _Plan:
            strategy = "offset"
            partitions = (1, 2, 3, 4)
            total_rows = 40

        class _Request:
            diagnostics = request_diagnostics

        _log_completion(
            _Resource(),
            _Request(),
            _Plan(),
            pd.DataFrame({"id": [1]}),
            perf_counter(),
            (gate_key, baseline),
        )
        return recorded[-1] if recorded else ""

    def test_reports_gate_wait_when_fetches_ran(self) -> None:
        with _timed_fetch_gate("gateway-ran", 4):
            pass

        line = self._run(True, "gateway-ran")
        assert "gate_waits=0/1" in line, line
        assert "gate_wait_total=" in line

    def test_omits_gate_wait_when_no_fetch_ran_here(self) -> None:
        line = self._run(True, "gateway-never-fetched")
        assert "load completed" in line
        assert "gate_wait" not in line, line
