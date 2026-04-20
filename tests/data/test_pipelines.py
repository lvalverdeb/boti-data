from __future__ import annotations

import datetime as dt

import dask.dataframe as dd
import pandas as pd
import pytest
from sqlalchemy import Date, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import (
    AsyncFrameEnricher,
    AttachmentSpec,
    CsvSink,
    CsvSinkConfig,
    DataHelper,
    HybridDataset,
    JsonlSink,
    ParquetMaterializationResult,
    ParquetPipeline,
    ParquetReader,
    SinkPipeline,
    SinkWriteResult,
)


class Base(DeclarativeBase):
    pass


class SourceEvent(Base):
    __tablename__ = "source_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(32))


class HistoricalEvent(Base):
    __tablename__ = "historical_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(32))


class LiveEvent(Base):
    __tablename__ = "live_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(32))


def _build_source_helper(tmp_path) -> DataHelper:
    db_path = tmp_path / "pipeline_source.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    SourceEvent(id=1, event_date=dt.date(2026, 4, 15), status="active"),
                    SourceEvent(id=2, event_date=dt.date(2026, 4, 16), status="inactive"),
                    SourceEvent(id=3, event_date=dt.date(2026, 4, 17), status="active"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    return DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="source_events",
    )


def _build_hybrid_dataset(tmp_path) -> HybridDataset:
    db_path = tmp_path / "pipeline_hybrid.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    HistoricalEvent(id=1, event_date=dt.date(2026, 4, 15), status="hist"),
                    HistoricalEvent(id=2, event_date=dt.date(2026, 4, 17), status="hist"),
                    LiveEvent(id=10, event_date=dt.date(2026, 4, 18), status="live"),
                    LiveEvent(id=11, event_date=dt.date(2026, 4, 20), status="live"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    historical_helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="historical_events",
    )
    live_helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="live_events",
    )
    return HybridDataset(
        historical_helper,
        live_helper,
        date_field="event_date",
        split_date="2026-04-18",
    )


def test_parquet_pipeline_materializes_datahelper_and_reloads(temp_project_root):
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


def test_parquet_pipeline_materialize_returns_result_with_optional_reload(temp_project_root):
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
        computed = reloaded.frame.compute().sort_values("id").reset_index(drop=True)
    finally:
        pipeline.close()

    assert computed["id"].tolist() == [3]
    assert computed["partition_date"].tolist() == ["2026-04-17"]


@pytest.mark.asyncio
async def test_parquet_pipeline_materializes_hybrid_dataset_and_reloads(temp_project_root):
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
async def test_parquet_pipeline_amaterialize_supports_reload_and_returns_result(temp_project_root):
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
        computed = result.frame.compute().sort_values("id").reset_index(drop=True)
    finally:
        await pipeline.aclose()

    assert computed["id"].tolist() == [10]
    assert computed["status"].tolist() == ["live"]


def test_parquet_pipeline_rejects_file_destination(temp_project_root):
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


def test_parquet_pipeline_rejects_eager_materialization_requests(temp_project_root):
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


def test_sink_pipeline_writes_csv_dataset_with_partition_derivation(temp_project_root):
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
async def test_sink_pipeline_awrite_supports_hybrid_dataset_csv_sink(temp_project_root):
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


def test_sink_pipeline_can_create_named_sink_via_registry(temp_project_root):
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


def test_sink_pipeline_writes_jsonl_dataset(temp_project_root):
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


def test_sink_pipeline_applies_enricher_before_write(temp_project_root):
    helper = _build_source_helper(temp_project_root)

    async def attachment_fn(ids):
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


