"""
Chunked __in list splitting on the async aload() path: explicit
in_chunk_size/in_chunk_concurrency, auto filter-hint policy, concurrency
caps, and nested filter payloads.

Split out of test_field_map_gateway.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import SecretStr
from sqlalchemy import select

from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.gateway import DataGateway, GatewayPolicies
from boti_data.gateway.frame_convert import FrameResult
from boti_data.gateway.planning import InChunkPolicy

from .conftest import LegacyProduct, _legacy_gw

# ---------------------------------------------------------------------------
# Item 5: chunked __in list splitting
# ---------------------------------------------------------------------------


class _ConcurrencyTracker:
    """Tracks the in-flight call count and its high-water mark."""

    def __init__(self) -> None:
        self.current = 0
        self.max_seen = 0

    def enter(self) -> None:
        self.current += 1
        self.max_seen = max(self.max_seen, self.current)

    def exit(self) -> None:
        self.current -= 1


def _wrap_chunked_in_load_capturing(original: Any, captured: dict[str, Any]) -> Any:
    """Wrap ``original`` (a bound ``_chunked_in_load``) to record its chunk_size/
    max_concurrency args before delegating, for tests asserting on auto-hint
    resolution."""

    async def wrapped(
        execute_fn, chunk_size, options, *, return_type, max_concurrency=None
    ) -> FrameResult:
        captured["chunk_size"] = chunk_size
        captured["max_concurrency"] = max_concurrency
        return await original(
            execute_fn,
            chunk_size,
            options,
            return_type=return_type,
            max_concurrency=max_concurrency,
        )

    return wrapped


def test_aload_chunked_in_splits_and_concatenates(legacy_dsn) -> None:
    """Oversized __in list is chunked; all rows are returned."""
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    # All 3 rows have global_track_id in [10, 20, 30]; chunk_size=1 forces 3 tasks
    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[10, 20, 30],
                in_chunk_size=1,
                as_pandas=True,
            )
        )
        assert len(df) == 3
        assert set(df["global_track_id"].tolist()) == {10, 20, 30}
    finally:
        gw.close()


def test_aload_chunked_in_no_split_when_within_limit(legacy_dsn) -> None:
    """List within chunk_size limit is sent as a single query."""
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[10, 20, 30],
                in_chunk_size=900,
                as_pandas=True,
            )
        )
        assert len(df) == 3
    finally:
        gw.close()


def test_aload_chunked_in_empty_result(legacy_dsn) -> None:
    """Chunking with values that match nothing returns an empty result."""
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[999, 888],
                in_chunk_size=1,
                as_pandas=True,
            )
        )
        assert len(df) == 0
    finally:
        gw.close()


def test_aload_chunked_in_supports_pandas_index_values(legacy_dsn) -> None:
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=pd.Index([10, 20, 30]),
                in_chunk_size=1,
                as_pandas=True,
            )
        )
        assert len(df) == 3
        assert set(df["global_track_id"].tolist()) == {10, 20, 30}
    finally:
        gw.close()


def test_aload_chunked_in_preserves_other_filters(legacy_dsn) -> None:
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[10, 20, 30],
                product_type_id=1,
                in_chunk_size=1,
                in_chunk_concurrency=2,
                as_pandas=True,
            )
        )
        assert set(df["global_track_id"].tolist()) == {10, 20}
        assert set(df["product_type_id"].tolist()) == {1}
    finally:
        gw.close()


def test_aload_chunked_in_honors_concurrency_cap() -> None:
    import asyncio

    gw = DataGateway(
        SqlDatabaseConfig(
            connection_url="sqlite:///:memory:",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
        )
    )

    tracker = _ConcurrencyTracker()

    async def execute_fn(**opts) -> pd.DataFrame:
        tracker.enter()
        await asyncio.sleep(0.01)
        tracker.exit()
        return pd.DataFrame({"value": list(opts["field__in"])})

    try:
        result = asyncio.run(
            gw._chunked_in_load(
                execute_fn,
                1,
                {"field__in": [1, 2, 3, 4]},
                return_type="pandas",
                max_concurrency=2,
            )
        )
        assert tracker.max_seen == 2
        assert result["value"].tolist() == [1, 2, 3, 4]
    finally:
        gw.close()


def test_aload_chunked_in_accepts_public_concurrency_option(legacy_dsn) -> None:
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[10, 20, 30],
                in_chunk_size=1,
                in_chunk_concurrency=2,
                as_pandas=True,
            )
        )
        assert len(df) == 3
        assert set(df["global_track_id"].tolist()) == {10, 20, 30}
    finally:
        gw.close()


def test_aload_chunked_in_auto_uses_filter_hints(legacy_dsn, monkeypatch) -> None:
    import asyncio

    gw = _legacy_gw(
        legacy_dsn,
        policies=GatewayPolicies(
            in_chunk_policy=InChunkPolicy(eager_auto_min_values=1, eager_auto_concurrency=2),
        ),
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        gw, "_chunked_in_load", _wrap_chunked_in_load_capturing(gw._chunked_in_load, captured)
    )

    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[10, 20, 30],
                as_pandas=True,
            )
        )
        assert len(df) == 3
        assert captured == {"chunk_size": 1, "max_concurrency": 2}
    finally:
        gw.close()


def test_aload_chunked_in_off_disables_auto_hint(legacy_dsn, monkeypatch) -> None:
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        gw, "_chunked_in_load", _wrap_chunked_in_load_capturing(gw._chunked_in_load, captured)
    )

    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[10, 20, 30],
                in_chunk_strategy="off",
                as_pandas=True,
            )
        )
        assert len(df) == 3
        assert captured == {"chunk_size": 0, "max_concurrency": None}
    finally:
        gw.close()


def test_aload_chunked_in_supports_nested_filter_payloads(legacy_dsn) -> None:
    import asyncio

    gw = DataGateway(SqlDatabaseConfig(connection_url=SecretStr(legacy_dsn), query_only=False))
    try:
        statement = select(LegacyProduct)
        df = asyncio.run(
            gw.aload(
                statement=statement,
                model=LegacyProduct,
                filters={"id_track_global__in": [10, 20, 30]},
                in_chunk_size=1,
                in_chunk_concurrency=2,
                as_pandas=True,
            )
        )
        assert set(df["id_track_global"].tolist()) == {10, 20, 30}
    finally:
        gw.close()
