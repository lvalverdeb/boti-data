"""Tests for DataHelper.load_incremental integration with watermarks, plus
related edge cases (missing table, alternate engine views).

Split out of test_watermark.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import datetime as dt
import time

from boti_data import DataHelper, FileWatermarkStore, IncrementalResult

from ._watermark_shared import IncEvent, _create_event_db

# ---------------------------------------------------------------------------
# Integration: DataHelper.load_incremental with SQLite backend
# ---------------------------------------------------------------------------


def test_incremental_first_run_full_load(tmp_path) -> None:
    dsn = _create_event_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
            {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
            {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "c"},
        ],
    )
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="inc_events",
    ) as helper:
        result = helper.pandas.load_incremental(
            watermark_field="event_date",
            watermark_source="test",
            watermark_store=store,
        )
    assert result.records_loaded == 3
    assert result.previous_watermark is None
    assert result.current_watermark is not None
    assert str(result.current_watermark).startswith("2026-05-03")
    assert result.watermark_committed is True
    persisted = store.read(source="test")
    assert persisted is not None
    assert str(persisted).startswith("2026-05-03")


def test_incremental_first_run_with_initial_value(tmp_path) -> None:
    dsn = _create_event_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
            {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
            {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "c"},
        ],
    )
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="inc_events",
    ) as helper:
        result = helper.pandas.load_incremental(
            watermark_field="event_date",
            watermark_source="test",
            watermark_store=store,
            initial_value=dt.date(2026, 5, 2),
        )
    # Only rows with date > 2026-05-02
    assert result.records_loaded == 1
    assert result.current_watermark is not None
    assert str(result.current_watermark).startswith("2026-05-03")


def test_incremental_second_run_delta(tmp_path) -> None:
    dsn = _create_event_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
            {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
            {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "c"},
        ],
    )
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.write(source="test", value=dt.date(2026, 5, 2))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="inc_events",
    ) as helper:
        result = helper.pandas.load_incremental(
            watermark_field="event_date",
            watermark_source="test",
            watermark_store=store,
        )
    # Only rows with date > 2026-05-02
    assert result.records_loaded == 1
    assert result.current_watermark is not None
    assert str(result.current_watermark).startswith("2026-05-03")
    assert str(result.previous_watermark) == "2026-05-02"


def test_incremental_no_new_data(tmp_path) -> None:
    dsn = _create_event_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        ],
    )
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.write(source="test", value=dt.date(2026, 5, 10))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="inc_events",
    ) as helper:
        start = time.monotonic()
        result = helper.pandas.load_incremental(
            watermark_field="event_date",
            watermark_source="test",
            watermark_store=store,
        )
        elapsed = time.monotonic() - start
    assert result.records_loaded == 0
    assert result.current_watermark is None
    assert result.watermark_committed is False
    assert elapsed < 2.0, f"0-row delta took {elapsed:.3f}s (threshold: 2.0s)"


def test_incremental_commit_false(tmp_path) -> None:
    dsn = _create_event_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        ],
    )
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="inc_events",
    ) as helper:
        result = helper.pandas.load_incremental(
            watermark_field="event_date",
            watermark_source="test",
            watermark_store=store,
            commit_on_success=False,
        )
    assert result.records_loaded == 1
    assert result.watermark_committed is False
    assert store.read(source="test") is None


def test_incremental_watermark_advances(tmp_path) -> None:
    """After loading rows 1-3, watermark advances. After loading 4-5, watermark advances again."""
    dsn = _create_event_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
            {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
            {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "c"},
            {"id": 4, "event_date": dt.date(2026, 5, 4), "status": "d"},
            {"id": 5, "event_date": dt.date(2026, 5, 5), "status": "e"},
        ],
    )
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="inc_events",
    ) as helper:
        first = helper.pandas.load_incremental(
            watermark_field="event_date",
            watermark_source="test",
            watermark_store=store,
            initial_value=dt.date(2026, 5, 3),
        )
        assert first.records_loaded == 2  # > 2026-05-03 → ids 4,5
        assert first.current_watermark is not None
        assert str(first.current_watermark).startswith("2026-05-05")

        second = helper.pandas.load_incremental(
            watermark_field="event_date",
            watermark_source="test",
            watermark_store=store,
        )
        assert second.records_loaded == 0  # no rows > 2026-05-05


def test_incremental_async(tmp_path) -> None:
    dsn = _create_event_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
            {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
        ],
    )
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.write(source="test", value=dt.date(2026, 5, 1))

    async def run() -> IncrementalResult:
        async with DataHelper(
            backend="sqlalchemy",
            connection_url=dsn,
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
            table="inc_events",
        ) as helper:
            return await helper.pandas.aload_incremental(
                watermark_field="event_date",
                watermark_source="test",
                watermark_store=store,
            )

    import asyncio

    result = asyncio.run(run())
    assert result.records_loaded == 1  # only id=2 (date > 2026-05-01)
    assert result.current_watermark is not None
    assert str(result.current_watermark).startswith("2026-05-02")


def test_incremental_dask_engine_view(tmp_path) -> None:
    dsn = _create_event_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
            {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
            {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "c"},
        ],
    )
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="inc_events",
    ) as helper:
        result = helper.dask.load_incremental(
            watermark_field="id",
            watermark_source="test",
            watermark_store=store,
        )
    assert result.records_loaded == 3
    assert result.current_watermark == 3


def test_incremental_with_explicit_filters(tmp_path) -> None:
    """Explicit filters passed alongside incremental filters are merged."""
    dsn = _create_event_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "active"},
            {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "inactive"},
            {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "active"},
        ],
    )
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.write(source="test", value=dt.date(2026, 5, 1))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="inc_events",
    ) as helper:
        result = helper.pandas.load_incremental(
            watermark_field="event_date",
            watermark_source="test",
            watermark_store=store,
            filters={"status__exact": "active"},
        )
    # Only rows with date > 2026-05-01 AND status=active
    assert result.records_loaded == 1


# ---------------------------------------------------------------------------
# Edge cases: error handling, clean state
# ---------------------------------------------------------------------------


def test_incremental_without_table_specified(tmp_path) -> None:
    """Works even when the helper has no 'table' set (falls back to 'default')."""
    dsn = _create_event_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        ],
    )
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    ) as helper:
        from sqlalchemy import select

        stmt = select(IncEvent)
        result = helper.pandas.load_incremental(
            watermark_field="event_date",
            watermark_source="no_table_test",
            watermark_store=store,
            statement=stmt,
            model=IncEvent,
        )
    assert result.records_loaded == 1


def test_incremental_polars_engine_view(tmp_path) -> None:
    """load_incremental works through the polars engine view."""
    dsn = _create_event_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
            {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
        ],
    )
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="inc_events",
    ) as helper:
        result = helper.polars.load_incremental(
            watermark_field="event_date",
            watermark_source="polars_test",
            watermark_store=store,
        )
    assert result.records_loaded == 2
    assert result.watermark_committed is True
