"""
Tests for SinkPipeline/CsvSink/JsonlSink write behavior and the
partition-derivation helper they share.

Split out of test_pipelines.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import pytest

from boti_data import (
    AsyncFrameEnricher,
    AttachmentSpec,
    CsvSink,
    CsvSinkConfig,
    JsonlSink,
    SinkPipeline,
    SinkWriteResult,
)

from ._pipelines_shared import _build_hybrid_dataset, _build_source_helper


def test_sink_pipeline_writes_csv_dataset_with_partition_derivation(temp_project_root) -> None:
    helper = _build_source_helper(temp_project_root)
    sink = CsvSink(
        CsvSinkConfig(
            storage_path=str(temp_project_root / "csv_sink_output"),
            partition_on=["partition_date"],
            project_root=temp_project_root,
        )
    )
    pipeline = SinkPipeline(helper, sink, date_field="event_date")
    try:
        result = pipeline.write(filters={"status__exact": "active"})
    finally:
        pipeline.close()

    assert isinstance(result, SinkWriteResult)
    assert result.path == str((temp_project_root / "csv_sink_output").resolve())
    assert any(path.endswith(".csv") for path in result.files)
    assert (temp_project_root / "csv_sink_output" / "partition_date=2026-04-15").exists()
    assert (temp_project_root / "csv_sink_output" / "partition_date=2026-04-17").exists()


@pytest.mark.asyncio
async def test_sink_pipeline_awrite_supports_hybrid_dataset_csv_sink(temp_project_root) -> None:
    dataset = _build_hybrid_dataset(temp_project_root)
    sink = CsvSink(
        {
            "storage_path": str(temp_project_root / "csv_sink_hybrid_output"),
            "partition_on": ["partition_date"],
            "project_root": temp_project_root,
        }
    )
    pipeline = SinkPipeline(dataset, sink, date_field="event_date")
    try:
        result = await pipeline.awrite(start="2026-04-15", end="2026-04-20")
    finally:
        await pipeline.aclose()

    assert isinstance(result, SinkWriteResult)
    assert result.path == str((temp_project_root / "csv_sink_hybrid_output").resolve())
    assert len(result.files) >= 2
    assert (temp_project_root / "csv_sink_hybrid_output" / "partition_date=2026-04-18").exists()


def test_sink_pipeline_can_create_named_sink_via_registry(temp_project_root) -> None:
    helper = _build_source_helper(temp_project_root)
    pipeline = SinkPipeline(
        helper,
        "csv",
        sink_config={
            "storage_path": str(temp_project_root / "csv_named_sink_output"),
            "partition_on": ["partition_date"],
            "project_root": temp_project_root,
        },
        date_field="event_date",
    )
    try:
        result = pipeline.write(filters={"status__exact": "active"})
    finally:
        pipeline.close()

    assert isinstance(result, SinkWriteResult)
    assert result.files


def test_sink_pipeline_writes_jsonl_dataset(temp_project_root) -> None:
    helper = _build_source_helper(temp_project_root)
    sink = JsonlSink(
        {
            "storage_path": str(temp_project_root / "jsonl_sink_output"),
            "partition_on": ["partition_date"],
            "project_root": temp_project_root,
        }
    )
    pipeline = SinkPipeline(helper, sink, date_field="event_date")
    try:
        result = pipeline.write(filters={"status__exact": "active"})
    finally:
        pipeline.close()

    assert isinstance(result, SinkWriteResult)
    assert any(path.endswith(".jsonl") for path in result.files)


def test_sink_pipeline_applies_enricher_before_write(temp_project_root) -> None:
    helper = _build_source_helper(temp_project_root)

    async def attachment_fn(ids) -> pd.DataFrame:
        return pd.DataFrame({"id": ids, "label": [f"id_{value}" for value in ids]})

    enricher = AsyncFrameEnricher(
        [
            AttachmentSpec(
                key="labels",
                required_cols={"id"},
                attachment_fn=attachment_fn,
                col_to_kwarg={"id": "ids"},
                left_on=["id"],
                right_on=["id"],
                drop_cols=[],
            )
        ]
    )

    sink = CsvSink(
        {
            "storage_path": str(temp_project_root / "csv_enriched_output"),
            "partition_on": ["partition_date"],
            "project_root": temp_project_root,
        }
    )
    pipeline = SinkPipeline(helper, sink, date_field="event_date", enricher=enricher)
    try:
        result = pipeline.write(filters={"status__exact": "active"}, enrich_cols=["labels"])
    finally:
        pipeline.close()

    sample = pd.read_csv(str(result.files[0]))
    assert "label" in sample.columns


def test_csv_sink_overwrite_preserves_previous_output_when_write_fails(temp_project_root) -> None:
    """A failing compute during overwrite must not destroy the existing dataset."""
    target = temp_project_root / "csv_staging_output"
    sink = CsvSink(
        CsvSinkConfig(
            storage_path=str(target),
            project_root=temp_project_root,
        )
    )
    try:
        first = pd.DataFrame({"id": [1, 2], "status": ["a", "b"]})
        sink.write(first)
        assert target.exists()
        before = sorted(p.name for p in target.iterdir())

        def _explode(df: pd.DataFrame) -> pd.DataFrame:
            raise RuntimeError("simulated compute failure")

        broken = dd.from_pandas(pd.DataFrame({"id": [3], "status": ["c"]}), npartitions=1)
        broken = broken.map_partitions(
            _explode,
            meta=pd.DataFrame(
                {"id": pd.Series(dtype="int64"), "status": pd.Series(dtype="object")}
            ),
        )

        with pytest.raises(Exception):
            sink.write(broken)

        # Previous output survives the failed overwrite.
        assert target.exists()
        assert sorted(p.name for p in target.iterdir()) == before
        survivor = pd.read_csv(next(target.glob("*.csv")))
        assert sorted(survivor["id"].tolist()) == [1, 2]

        # A subsequent successful overwrite cleans up and replaces.
        second = pd.DataFrame({"id": [9], "status": ["z"]})
        result = sink.write(second)
        replaced = pd.read_csv(str(result.files[0]))
        assert replaced["id"].tolist() == [9]
        assert not (temp_project_root / "csv_staging_output.staging").exists()
    finally:
        sink.close()


def test_prepare_partitioned_frame_preserves_explicit_bool_meta_after_assign() -> None:
    """A partition_date derivation must not corrupt unrelated bool column meta.

    Regression test for a dask-expr defect where .assign() re-derives _meta for
    the whole frame, silently dropping an explicit per-column dtype hint set by
    an upstream map_partitions(meta=(col, bool)) cast and falling back to
    generic `object` dtype -- which later crashes to_parquet's schema inference.
    """
    from boti_data.pipelines.sinks import prepare_partitioned_frame

    pdf = pd.DataFrame(
        {
            "confirmation_dt": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "dispatched": [True, False],
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=1)
    # Simulate the upstream fix_data() cast that pins an explicit bool meta hint.
    ddf["dispatched"] = ddf["dispatched"].map_partitions(
        lambda s: s.astype(bool), meta=("dispatched", bool)
    )
    assert ddf.dtypes["dispatched"] == bool  # noqa: E721 -- numpy dtype equality, not type identity

    result = prepare_partitioned_frame(
        ddf,
        partition_on=["partition_date"],
        date_field="confirmation_dt",
        sink_name="TestSink",
    )

    assert result.dtypes["dispatched"] == bool  # noqa: E721
    assert type(result._meta_nonempty["dispatched"].iloc[0]) is not object
