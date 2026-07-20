"""
Tests for ParquetPipeline materialization/reload behavior.

Split out of test_pipelines.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import dask.dataframe as dd
import pytest

from boti_data import ParquetMaterializationResult, ParquetPipeline, ParquetReader

from ._pipelines_shared import _build_hybrid_dataset, _build_source_helper


def test_parquet_pipeline_materializes_datahelper_and_reloads(temp_project_root) -> None:
    helper = _build_source_helper(temp_project_root)
    pipeline = ParquetPipeline(
        helper,
        {
            "storage_path": str(temp_project_root / "pipeline_output"),
            "project_root": temp_project_root,
            "partition_on": ["partition_date"],
        },
        date_field="event_date",
    )
    try:
        output_path = pipeline.to_parquet(filters={"status__exact": "active"})
        frame = pipeline.from_parquet(filters={"status__exact": "active"})
        computed = frame.compute().sort_values("id").reset_index(drop=True)
    finally:
        pipeline.close()

    assert output_path == str(temp_project_root / "pipeline_output")
    assert isinstance(frame, dd.DataFrame)
    assert computed["id"].tolist() == [1, 3]
    assert computed["partition_date"].tolist() == ["2026-04-15", "2026-04-17"]
    assert (temp_project_root / "pipeline_output" / "partition_date=2026-04-15").exists()


def test_parquet_pipeline_materialize_returns_result_with_optional_reload(
    temp_project_root,
) -> None:
    helper = _build_source_helper(temp_project_root)
    pipeline = ParquetPipeline(
        helper,
        {
            "storage_path": str(temp_project_root / "pipeline_materialize_output"),
            "project_root": temp_project_root,
            "partition_on": ["partition_date"],
        },
        date_field="event_date",
    )
    try:
        write_only = pipeline.materialize(filters={"status__exact": "active"})
        reloaded = pipeline.materialize(
            reload=True,
            filters={"status__exact": "active"},
            reload_options={"filters": {"partition_date__exact": "2026-04-17"}},
        )
        assert isinstance(write_only, ParquetMaterializationResult)
        assert write_only.path == str(temp_project_root / "pipeline_materialize_output")
        assert write_only.frame is None
        assert write_only.reloaded is False

        assert isinstance(reloaded, ParquetMaterializationResult)
        assert reloaded.reloaded is True
        assert isinstance(reloaded.frame, dd.DataFrame)
        reloaded_computed = reloaded.frame.compute()
        computed = reloaded_computed.sort_values("id").reset_index(drop=True)
    finally:
        pipeline.close()

    assert computed["id"].tolist() == [3]
    assert computed["partition_date"].tolist() == ["2026-04-17"]


@pytest.mark.asyncio
async def test_parquet_pipeline_materializes_hybrid_dataset_and_reloads(temp_project_root) -> None:
    dataset = _build_hybrid_dataset(temp_project_root)
    pipeline = ParquetPipeline(
        dataset,
        {
            "storage_path": str(temp_project_root / "hybrid_pipeline_output"),
            "project_root": temp_project_root,
            "partition_on": ["partition_date"],
        },
        date_field="event_date",
    )
    try:
        output_path = await pipeline.ato_parquet(start="2026-04-15", end="2026-04-20")
        frame = await pipeline.afrom_parquet()
        computed = frame.compute().sort_values("id").reset_index(drop=True)
    finally:
        await pipeline.aclose()

    assert output_path == str(temp_project_root / "hybrid_pipeline_output")
    assert isinstance(frame, dd.DataFrame)
    assert computed["id"].tolist() == [1, 2, 10, 11]
    assert computed["status"].tolist() == ["hist", "hist", "live", "live"]


@pytest.mark.asyncio
async def test_parquet_pipeline_amaterialize_supports_reload_and_returns_result(
    temp_project_root,
) -> None:
    dataset = _build_hybrid_dataset(temp_project_root)
    pipeline = ParquetPipeline(
        dataset,
        {
            "storage_path": str(temp_project_root / "hybrid_materialize_output"),
            "project_root": temp_project_root,
            "partition_on": ["partition_date"],
        },
        date_field="event_date",
    )
    try:
        result = await pipeline.amaterialize(
            start="2026-04-15",
            end="2026-04-20",
            reload=True,
            reload_options={"filters": {"partition_date__exact": "2026-04-18"}},
        )
        assert isinstance(result, ParquetMaterializationResult)
        assert result.path == str(temp_project_root / "hybrid_materialize_output")
        assert result.reloaded is True
        assert isinstance(result.frame, dd.DataFrame)
        result_computed = result.frame.compute()
        computed = result_computed.sort_values("id").reset_index(drop=True)
    finally:
        await pipeline.aclose()

    assert computed["id"].tolist() == [10]
    assert computed["status"].tolist() == ["live"]


def test_parquet_pipeline_rejects_file_destination(temp_project_root) -> None:
    helper = _build_source_helper(temp_project_root)
    reader = ParquetReader(
        storage_path=str(temp_project_root / "single_file_output"),
        parquet_filename="events",
        project_root=temp_project_root,
    )
    try:
        with pytest.raises(ValueError, match="parquet_filename is not supported"):
            ParquetPipeline(helper, reader, date_field="event_date")
    finally:
        reader.close()
        helper.close()


def test_parquet_pipeline_rejects_eager_materialization_requests(temp_project_root) -> None:
    helper = _build_source_helper(temp_project_root)
    pipeline = ParquetPipeline(
        helper,
        {
            "storage_path": str(temp_project_root / "pipeline_invalid_options"),
            "project_root": temp_project_root,
            "partition_on": ["partition_date"],
        },
        date_field="event_date",
    )
    try:
        with pytest.raises(ValueError, match="return_type='dask'"):
            pipeline.to_parquet(return_type="pandas")
    finally:
        pipeline.close()
