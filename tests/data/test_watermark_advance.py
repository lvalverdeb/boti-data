"""Tests for advance_watermark, IncrementalResult, and ParquetPipeline.incremental
integration with watermarks.

Split out of test_watermark.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import datetime as dt
import math

import dask.dataframe as dd
import pandas as pd

from boti_data import (
    DataHelper,
    FileWatermarkStore,
    IncrementalResult,
    ParquetPipeline,
    advance_watermark,
)

from ._watermark_shared import _create_pipeline_db

# ---------------------------------------------------------------------------
# advance_watermark
# ---------------------------------------------------------------------------


def test_advance_watermark_datetime() -> None:
    df = pd.DataFrame({"ts": pd.to_datetime(["2026-05-01", "2026-05-03", "2026-05-02"])})
    result = advance_watermark(df, watermark_field="ts")
    assert result == pd.Timestamp("2026-05-03")


def test_advance_watermark_integer() -> None:
    df = pd.DataFrame({"id": [1, 5, 3, 9, 2]})
    result = advance_watermark(df, watermark_field="id")
    assert result == 9


def test_advance_watermark_empty_frame() -> None:
    df = pd.DataFrame({"val": []})
    result = advance_watermark(df, watermark_field="val")
    assert result is None


def test_advance_watermark_all_null() -> None:
    df = pd.DataFrame({"val": [None, None]})
    result = advance_watermark(df, watermark_field="val")
    assert result is None or (isinstance(result, float) and math.isnan(result))


def test_advance_watermark_dask() -> None:
    pdf = pd.DataFrame({"id": [10, 20, 30]})
    ddf = dd.from_pandas(pdf, npartitions=2)
    result = advance_watermark(ddf, watermark_field="id")
    assert result == 30


def test_advance_watermark_empty_dask() -> None:
    pdf = pd.DataFrame({"id": []})
    ddf = dd.from_pandas(pdf, npartitions=1)
    result = advance_watermark(ddf, watermark_field="id")
    assert result is None


def test_advance_watermark_string() -> None:
    df = pd.DataFrame({"code": ["a", "z", "m"]})
    result = advance_watermark(df, watermark_field="code")
    assert result == "z"


# ---------------------------------------------------------------------------
# IncrementalResult
# ---------------------------------------------------------------------------


def test_incremental_result_bool_true() -> None:
    r = IncrementalResult(
        frame=pd.DataFrame({"a": [1]}),
        watermark_field="a",
        previous_watermark=None,
        current_watermark=1,
        records_loaded=1,
        watermark_committed=True,
    )
    assert bool(r) is True


def test_incremental_result_bool_false() -> None:
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
# Integration: ParquetPipeline.incremental
# ---------------------------------------------------------------------------


def test_parquet_pipeline_incremental(tmp_path) -> None:
    dsn = _create_pipeline_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
            {"id": 2, "event_date": dt.date(2026, 5, 2), "status": "b"},
            {"id": 3, "event_date": dt.date(2026, 5, 3), "status": "c"},
        ],
    )
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
        try:
            r1 = pipeline.materialize(reload=True)
            assert r1.path is not None
            loaded = int(r1.frame.compute().shape[0]) if r1.frame is not None else 0
            assert loaded == 3
            assert store.read(source="pipeline") is not None

            # Second run — no new data
            r2 = pipeline.materialize(reload=True)
            loaded2 = int(r2.frame.compute().shape[0]) if r2.frame is not None else 0
            assert loaded2 == 0
        finally:
            pipeline.close()


def test_parquet_pipeline_incremental_no_reload(tmp_path) -> None:
    """Without reload=True, the watermark is still updated."""
    dsn = _create_pipeline_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        ],
    )
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
        try:
            # First run without reload — watermark still committed
            pipeline.materialize(reload=False)
            assert store.read(source="pipeline2") is not None
        finally:
            pipeline.close()


def test_parquet_pipeline_incremental_custom_source_name(tmp_path) -> None:
    dsn = _create_pipeline_db(
        tmp_path,
        [
            {"id": 1, "event_date": dt.date(2026, 5, 1), "status": "a"},
        ],
    )
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
        try:
            pipeline.materialize()
            assert store.read(source="custom_pipeline") is not None
        finally:
            pipeline.close()
