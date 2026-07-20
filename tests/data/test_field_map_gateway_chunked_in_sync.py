"""
Chunked __in list splitting on the sync load() path: explicit
in_chunk_size/in_chunk_concurrency, auto filter-hint policy, and
concurrency caps.

Split out of test_field_map_gateway.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import threading
from typing import Any

import pandas as pd

from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.gateway import DataGateway, GatewayPolicies
from boti_data.gateway.frame_convert import FrameResult
from boti_data.gateway.planning import InChunkPolicy

from .conftest import _legacy_gw


class _ThreadSafeConcurrencyTracker:
    """Tracks the in-flight call count and its high-water mark across threads."""

    def __init__(self) -> None:
        self.current = 0
        self.max_seen = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)

    def exit(self) -> None:
        with self._lock:
            self.current -= 1


def _wrap_chunked_in_load_sync_capturing(original: Any, captured: dict[str, Any]) -> Any:
    """Wrap ``original`` (a bound ``_chunked_in_load_sync``) to record its
    chunk_size/max_concurrency args before delegating, for tests asserting on
    auto-hint resolution."""

    def wrapped(
        execute_fn, chunk_size, options, *, return_type, max_concurrency=None
    ) -> FrameResult:
        captured["chunk_size"] = chunk_size
        captured["max_concurrency"] = max_concurrency
        return original(
            execute_fn,
            chunk_size,
            options,
            return_type=return_type,
            max_concurrency=max_concurrency,
        )

    return wrapped


def test_load_chunked_in_accepts_public_concurrency_option(legacy_dsn) -> None:
    gw = _legacy_gw(legacy_dsn)
    try:
        df = gw.load(
            global_track_id__in=[10, 20, 30],
            in_chunk_size=1,
            in_chunk_concurrency=2,
            as_pandas=True,
        )
        assert len(df) == 3
        assert set(df["global_track_id"].tolist()) == {10, 20, 30}
    finally:
        gw.close()


def test_load_chunked_in_auto_uses_filter_hints(legacy_dsn, monkeypatch) -> None:
    gw = _legacy_gw(
        legacy_dsn,
        policies=GatewayPolicies(
            in_chunk_policy=InChunkPolicy(eager_auto_min_values=1, eager_auto_concurrency=2),
        ),
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        gw,
        "_chunked_in_load_sync",
        _wrap_chunked_in_load_sync_capturing(gw._chunked_in_load_sync, captured),
    )

    try:
        df = gw.load(global_track_id__in=[10, 20, 30], as_pandas=True)
        assert len(df) == 3
        assert captured == {"chunk_size": 1, "max_concurrency": 2}
    finally:
        gw.close()


def test_load_chunked_in_explicit_values_override_auto_hint(legacy_dsn, monkeypatch) -> None:
    gw = _legacy_gw(legacy_dsn)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        gw,
        "_chunked_in_load_sync",
        _wrap_chunked_in_load_sync_capturing(gw._chunked_in_load_sync, captured),
    )

    try:
        df = gw.load(
            global_track_id__in=[10, 20, 30],
            in_chunk_size=2,
            in_chunk_concurrency=1,
            as_pandas=True,
        )
        assert len(df) == 3
        assert captured == {"chunk_size": 2, "max_concurrency": 1}
    finally:
        gw.close()


def test_load_chunked_in_preserves_other_filters(legacy_dsn) -> None:
    gw = _legacy_gw(legacy_dsn)
    try:
        df = gw.load(
            global_track_id__in=[10, 20, 30],
            product_type_id=1,
            in_chunk_size=1,
            in_chunk_concurrency=2,
            as_pandas=True,
        )
        assert set(df["global_track_id"].tolist()) == {10, 20}
        assert set(df["product_type_id"].tolist()) == {1}
    finally:
        gw.close()


def test_load_chunked_in_honors_concurrency_cap_sync() -> None:
    import time

    gw = DataGateway(
        SqlDatabaseConfig(
            connection_url="sqlite:///:memory:",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
        )
    )

    tracker = _ThreadSafeConcurrencyTracker()

    def execute_fn(**opts) -> pd.DataFrame:
        tracker.enter()
        time.sleep(0.01)
        tracker.exit()
        return pd.DataFrame({"value": list(opts["field__in"])})

    try:
        result = gw._chunked_in_load_sync(
            execute_fn,
            1,
            {"field__in": [1, 2, 3, 4]},
            return_type="pandas",
            max_concurrency=2,
        )
        assert tracker.max_seen == 2
        assert result["value"].tolist() == [1, 2, 3, 4]
    finally:
        gw.close()
