from __future__ import annotations

from boti_data._optional import dataframes_required

# Everything re-exported here reaches dask/pandas/polars, so surface the
# missing-extra hint instead of a bare "No module named 'dask'".
with dataframes_required(__name__):
    from .base import SinkPipeline
    from .parquet_pipeline import ParquetMaterializationResult, ParquetPipeline
    from .registry import SinkRegistry, available_sinks, create_sink, register_sink
    from .sinks import (
        CsvSink,
        CsvSinkConfig,
        JsonlSink,
        JsonlSinkConfig,
        ParquetSink,
        SinkWriteResult,
        awrite_parquet,
        write_parquet,
    )

__all__ = [
    "CsvSink",
    "CsvSinkConfig",
    "JsonlSink",
    "JsonlSinkConfig",
    "ParquetMaterializationResult",
    "ParquetPipeline",
    "ParquetSink",
    "SinkRegistry",
    "SinkPipeline",
    "SinkWriteResult",
    "available_sinks",
    "awrite_parquet",
    "create_sink",
    "register_sink",
    "write_parquet",
]
