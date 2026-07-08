"""
Tests for watermark store, incremental filter/advance helpers, and
DataHelper / ParquetPipeline incremental loading integration.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import threading
import time

import dask.dataframe as dd
import pandas as pd
from sqlalchemy import Date, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import (
    DataHelper,
    FileWatermarkStore,
    IncrementalResult,
    ParquetPipeline,
    advance_watermark,
    build_incremental_filters,
)

# ---------------------------------------------------------------------------
# WatermarkStore
# ---------------------------------------------------------------------------


def test_file_store_read_write_roundtrip(tmp_path):
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.write(source="test", value="2026-05-10")
    assert store.read(source="test") == "2026-05-10"


def test_file_store_clear(tmp_path):
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.write(source="t1", value=100)
    store.write(source="t2", value=200)
    store.clear(source="t1")
    assert store.read(source="t1") is None
    assert store.read(source="t2") == 200


def test_file_store_missing_key(tmp_path):
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    assert store.read(source="nonexistent") is None


def test_file_store_nonexistent_file(tmp_path):
    store = FileWatermarkStore(str(tmp_path / "no_such_file.json"))
    assert store.read(source="x") is None


def test_file_store_overwrite_value(tmp_path):
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.write(source="k", value="first")
    store.write(source="k", value="second")
    assert store.read(source="k") == "second"


def test_file_store_multiple_sources(tmp_path):
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.write(source="a", value=1)
    store.write(source="b", value=2)
    store.write(source="c", value=3)
    assert store.read(source="a") == 1
    assert store.read(source="b") == 2
    assert store.read(source="c") == 3


def test_file_store_clear_nonexistent(tmp_path):
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    store.clear(source="nobody")  # should not raise


def test_file_store_concurrent_access(tmp_path):
    """Basic thread-safety smoke test."""
    store = FileWatermarkStore(str(tmp_path / "w.json"))
    errors: list[Exception] = []

    def writer() -> None:
        try:
            for i in range(50):
                store.write(source="concurrent", value=i)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert store.read(source="concurrent") is not None


def test_file_store_persists_to_disk(tmp_path):
    path = tmp_path / "persist.json"
    store = FileWatermarkStore(str(path))
    store.write(source="disk", value="hello")
    raw = json.loads(path.read_text())
    assert raw["disk"] == "hello"


# ---------------------------------------------------------------------------
# build_incremental_filters
# ---------------------------------------------------------------------------


def test_build_incremental_filters_gt():
    result = build_incremental_filters(
        watermark_field="updated_at",
        watermark_value="2026-05-01",
        operator="gt",
    )
    assert result == {"updated_at__gt": "2026-05-01"}


def test_build_incremental_filters_gte():
    result = build_incremental_filters(
        watermark_field="event_date",
        watermark_value="2026-05-01",
        operator="gte",
    )
    assert result == {"event_date__gte": "2026-05-01"}


def test_build_incremental_filters_default_operator():
    result = build_incremental_filters(
        watermark_field="id",
        watermark_value=100,
    )
    assert result == {"id__gt": 100}


# ---------------------------------------------------------------------------
# advance_watermark
# ---------------------------------------------------------------------------


def test_advance_watermark_datetime():
    df = pd.DataFrame({"ts": pd.to_datetime(["2026-05-01", "2026-05-03", "2026-05-02"])})
    result = advance_watermark(df, watermark_field="ts")
    assert result == pd.Timestamp("2026-05-03")


def test_advance_watermark_integer():
    df = pd.DataFrame({"id": [1, 5, 3, 9, 2]})
    result = advance_watermark(df, watermark_field="id")
    assert result == 9


def test_advance_watermark_empty_frame():
    df = pd.DataFrame({"val": []})
    result = advance_watermark(df, watermark_field="val")
    assert result is None


def test_advance_watermark_all_null():
    df = pd.DataFrame({"val": [None, None]})
    result = advance_watermark(df, watermark_field="val")
    assert result is None or (isinstance(result, float) and math.isnan(result))


def test_advance_watermark_dask():
    pdf = pd.DataFrame({"id": [10, 20, 30]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    result = advance_watermark(ddf, watermark_field="id")
    assert result == 30


def test_advance_watermark_empty_dask():
    pdf = pd.DataFrame({"id": []})
    ddf = dd.from_pandas(pdf, npartitions=1)
    result = advance_watermark(ddf, watermark_field="id")
    assert result is None


def test_advance_watermark_string():
    df = pd.DataFrame({"code": ["a", "z", "m"]})
    result = advance_watermark(df, watermark_field="code")
    assert result == "z"


# ---------------------------------------------------------------------------
# IncrementalResult
# ---------------------------------------------------------------------------


def test_incremental_result_bool_true():
    r = IncrementalResult(
        frame=pd.DataFrame({"a": [1]}),
        watermark_field="a",
        previous_watermark=None,
        current_watermark=1,
        records_loaded=1,
        watermark_committed=True,
    )
    assert bool(r) is True


def test_incremental_result_bool_false():
    r = IncrementalResult(
        frame=pd.DataFrame({"a": []}),
        watermark_field="a",
        previous_watermark=10,
        current_watermark=None,
        records_loaded=0,
        watermark_committed=False,
    )
    assert bool(r) is False


# ---------------------------------------------------------------------------
# Integration: DataHelper.load_incremental with SQLite backend
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class IncEvent(Base):
    __tablename__ = "inc_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(16))


def _create_event_db(tmp_path, rows: list[dict]) -> str:
    db_path = tmp_path / "inc_events.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([IncEvent(**r) for r in rows])
            session.commit()
    finally:
        engine.dispose()
    return f"sqlite:///{db_path}"


def test_incremental_first_run_full_load(tmp_path):
    dsn = _create_event_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
        {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "c"},
    ])
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


def test_incremental_first_run_with_initial_value(tmp_path):
    dsn = _create_event_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
        {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "c"},
    ])
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


def test_incremental_second_run_delta(tmp_path):
    dsn = _create_event_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
        {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "c"},
    ])
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


def test_incremental_no_new_data(tmp_path):
    dsn = _create_event_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
    ])
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


def test_incremental_commit_false(tmp_path):
    dsn = _create_event_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
    ])
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


def test_incremental_watermark_advances(tmp_path):
    """After loading rows 1-3, watermark advances. After loading 4-5, watermark advances again."""
    dsn = _create_event_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
        {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "c"},
        {"id": 4, "event_date": dt.date(2026, 5, 4), "status": "d"},
        {"id": 5, "event_date": dt.date(2026, 5, 5), "status": "e"},
    ])
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


def test_incremental_async(tmp_path):
    dsn = _create_event_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
    ])
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


def test_incremental_dask_engine_view(tmp_path):
    dsn = _create_event_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
        {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "c"},
    ])
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


def test_incremental_with_explicit_filters(tmp_path):
    """Explicit filters passed alongside incremental filters are merged."""
    dsn = _create_event_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "active"},
        {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "inactive"},
        {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "active"},
    ])
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
# Integration: ParquetPipeline.incremental
# ---------------------------------------------------------------------------


class PipelineEvent(Base):
    __tablename__ = "pipeline_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(16))


def _create_pipeline_db(tmp_path, rows: list[dict]) -> str:
    db_path = tmp_path / "pipeline.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all([PipelineEvent(**r) for r in rows])
            session.commit()
    finally:
        engine.dispose()
    return f"sqlite:///{db_path}"


def test_parquet_pipeline_incremental(tmp_path):
    dsn = _create_pipeline_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
        {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "c"},
    ])
    store = FileWatermarkStore(str(tmp_path / "pw.json"))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="pipeline_events",
    ) as helper:
        pipeline = ParquetPipeline.incremental(
            helper,
            {
                "backend": "parquet",
                "storage_path": str(tmp_path / "pipeline_ds"),
                "project_root": tmp_path,
            },
            watermark_field="event_date",
            watermark_store=store,
            watermark_source="pipeline",
            date_field="event_date",
        )
        r1 = pipeline.materialize(reload=True)
        assert r1.path is not None
        loaded = int(r1.frame.compute().shape[0]) if r1.frame is not None else 0
        assert loaded == 3
        assert store.read(source="pipeline") is not None

        # Second run — no new data
        r2 = pipeline.materialize(reload=True)
        loaded2 = int(r2.frame.compute().shape[0]) if r2.frame is not None else 0
        assert loaded2 == 0


def test_parquet_pipeline_incremental_no_reload(tmp_path):
    """Without reload=True, the watermark is still updated."""
    dsn = _create_pipeline_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
    ])
    store = FileWatermarkStore(str(tmp_path / "pw2.json"))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="pipeline_events",
    ) as helper:
        pipeline = ParquetPipeline.incremental(
            helper,
            {
                "backend": "parquet",
                "storage_path": str(tmp_path / "pipeline_ds2"),
                "project_root": tmp_path,
            },
            watermark_field="event_date",
            watermark_store=store,
            watermark_source="pipeline2",
            date_field="event_date",
        )
        # First run without reload — watermark still committed
        pipeline.materialize(reload=False)
        assert store.read(source="pipeline2") is not None


def test_parquet_pipeline_incremental_custom_source_name(tmp_path):
    dsn = _create_pipeline_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
    ])
    store = FileWatermarkStore(str(tmp_path / "pw3.json"))
    with DataHelper(
        backend="sqlalchemy",
        connection_url=dsn,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="pipeline_events",
    ) as helper:
        pipeline = ParquetPipeline.incremental(
            helper,
            {
                "backend": "parquet",
                "storage_path": str(tmp_path / "pipeline_ds3"),
                "project_root": tmp_path,
            },
            watermark_field="event_date",
            watermark_store=store,
            watermark_source="custom_pipeline",
            date_field="event_date",
        )
        pipeline.materialize()
        assert store.read(source="custom_pipeline") is not None


# ---------------------------------------------------------------------------
# Edge cases: error handling, clean state
# ---------------------------------------------------------------------------


def test_incremental_without_table_specified(tmp_path):
    """Works even when the helper has no 'table' set (falls back to 'default')."""
    dsn = _create_event_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
    ])
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


def test_incremental_polars_engine_view(tmp_path):
    """load_incremental works through the polars engine view."""
    dsn = _create_event_db(tmp_path, [
        {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
    ])
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
