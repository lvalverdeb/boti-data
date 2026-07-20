from __future__ import annotations

from .base import SinkPipeline
from .parquet_pipeline import ParquetMaterializationResult, ParquetPipeline
from .registry import SinkRegistry, available_sinks, create_sink, register_sink
from .sinks import CsvSink, CsvSinkConfig, JsonlSink, JsonlSinkConfig, ParquetSink, SinkWriteResult

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
    "create_sink",
    "register_sink",
]
