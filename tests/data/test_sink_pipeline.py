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
    ParquetDataConfig,
    SinkPipeline,
    SinkWriteResult,
    awrite_parquet,
    write_parquet,
)
from boti_data.pipelines.sinks_parquet import ParquetSink

from ._pipelines_shared import PrefixOnlyFakeFileSystem, _build_hybrid_dataset, _build_source_helper


class _StubLogger:
    def debug(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def error(self, *_args, **_kwargs) -> None:
        pass


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


def test_prepare_partitioned_frame_rejects_meta_corrupted_by_unrelated_reassignment() -> None:
    """Systemic guard for wishlist #2/#7: catch dask-expr meta corruption at write time.

    Unlike the regression above (corruption from this module's own internal
    partition_date .assign()), this reproduces the corruption happening
    entirely in *caller* code -- a bare dd.to_datetime() reassignment of one
    column silently turning an unrelated, already-correctly-typed bool
    column's meta into `object` -- with no partitioning involved at all. The
    sink must raise a clear, actionable error instead of shipping a
    corrupted frame through to a cryptic pyarrow failure later.
    """
    from boti_data.pipelines.sinks import prepare_partitioned_frame

    pdf = pd.DataFrame(
        {
            "dispatched": [True, False, True],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=1)
    ddf["dispatched"] = ddf["dispatched"].map_partitions(
        lambda s: s.astype(bool), meta=("dispatched", bool)
    )
    assert ddf.dtypes["dispatched"] == bool  # noqa: E721

    # Bare reassignment elsewhere in the pipeline -- the actual trigger from
    # the wishlist reproduction -- corrupts the unrelated bool column's meta.
    ddf["event_date"] = dd.to_datetime(ddf["event_date"])
    assert ddf.dtypes["dispatched"] != bool  # noqa: E721 -- meta already corrupted here

    with pytest.raises(ValueError, match="meta/real dtype mismatch"):
        prepare_partitioned_frame(
            ddf,
            partition_on=None,
            date_field=None,
            sink_name="TestSink",
        )


def test_write_parquet_closes_the_sink_automatically(temp_project_root, monkeypatch) -> None:
    """Wishlist #5: a one-shot write must not require an explicit `with ParquetSink(...):`."""
    close_calls: list[bool] = []
    original_cleanup = ParquetSink._cleanup

    def _tracking_cleanup(self) -> None:
        close_calls.append(True)
        original_cleanup(self)

    monkeypatch.setattr(ParquetSink, "_cleanup", _tracking_cleanup)

    frame = pd.DataFrame({"id": [1, 2], "partition_date": ["2026-04-15", "2026-04-16"]})
    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=_StubLogger(),
        parquet_storage_path=str(temp_project_root / "write_parquet_output"),
    )

    result = write_parquet(config, frame)

    assert close_calls == [True]
    assert isinstance(result, SinkWriteResult)
    assert (temp_project_root / "write_parquet_output" / "partition_date=2026-04-15").exists()
    assert (temp_project_root / "write_parquet_output" / "partition_date=2026-04-16").exists()


@pytest.mark.asyncio
async def test_awrite_parquet_closes_the_sink_automatically(temp_project_root, monkeypatch) -> None:
    close_calls: list[bool] = []
    original_acleanup = ParquetSink._acleanup

    async def _tracking_acleanup(self) -> None:
        close_calls.append(True)
        await original_acleanup(self)

    monkeypatch.setattr(ParquetSink, "_acleanup", _tracking_acleanup)

    frame = pd.DataFrame({"id": [1], "partition_date": ["2026-04-17"]})
    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=_StubLogger(),
        parquet_storage_path=str(temp_project_root / "awrite_parquet_output"),
    )

    result = await awrite_parquet(config, frame)

    assert close_calls == [True]
    assert isinstance(result, SinkWriteResult)
    assert (temp_project_root / "awrite_parquet_output" / "partition_date=2026-04-17").exists()


def test_write_parquet_rejects_materialized_file_destination(temp_project_root) -> None:
    """write_parquet() must surface ParquetSink's own destination validation, not swallow it."""
    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=_StubLogger(),
        parquet_storage_path=str(temp_project_root / "single_file"),
        parquet_filename="single_file",
    )

    with pytest.raises(ValueError, match="parquet_filename is not supported"):
        write_parquet(config, pd.DataFrame({"id": [1], "partition_date": ["2026-04-15"]}))


def test_write_with_staging_swap_avoids_expand_path_phantom_entry() -> None:
    """Regression for wishlist #1: MinIO/prefix-only-S3 overwrite swap.

    A single recursive ``fs.mv(staging_path, target_path, recursive=True)``
    re-expands the staging directory and, on a prefix-only backend, that
    expansion can surface a phantom entry for the bare directory prefix
    itself -- no object actually exists there -- which 404s the whole swap.
    ``_write_with_staging`` must move the already-enumerated file list one by
    one instead, so the phantom entry never gets a chance to appear.
    """
    from boti_data.pipelines.sinks_common import _write_with_staging

    fs = PrefixOnlyFakeFileSystem()
    target = "bucket/some/dir"
    fs.store[f"{target}/old_part.0.parquet"] = b"stale"

    # Sanity: the bug is real against this backend shape -- a bare recursive
    # mv over the same layout raises via the phantom expand_path entry.
    fs.store[f"{target}.staging/part.0.parquet"] = b"new"
    with pytest.raises(FileNotFoundError):
        fs.mv(f"{target}.staging", target, recursive=True)
    fs.store.pop(f"{target}.staging/part.0.parquet")

    def _write_fn(directory: str) -> list[str]:
        fs.store[f"{directory}/part.0.parquet"] = b"new"
        fs.store[f"{directory}/part.1.parquet"] = b"new2"
        return sorted(fs.find(directory))

    files = _write_with_staging(fs=fs, target_path=target, overwrite=True, write_fn=_write_fn)

    assert sorted(files) == [f"{target}/part.0.parquet", f"{target}/part.1.parquet"]
    assert fs.store == {
        f"{target}/part.0.parquet": b"new",
        f"{target}/part.1.parquet": b"new2",
    }
