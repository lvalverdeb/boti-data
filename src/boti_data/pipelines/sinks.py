"""Stable import surface for pipeline sinks.

Implementation lives in sinks_common.py (shared types/helpers/protocols),
sinks_text.py (Csv/Jsonl sinks), and sinks_parquet.py (ParquetSink), split
out purely for line-count headroom. Re-exported here so every existing
``from boti_data.pipelines.sinks import ...`` keeps working.
"""

from __future__ import annotations

from boti_data.pipelines.sinks_common import (
    FrameResult,
    ParquetDestination,
    PipelineSink,
    SinkWriteResult,
    prepare_partitioned_frame,
    to_dask_frame,
)
from boti_data.pipelines.sinks_parquet import ParquetSink, awrite_parquet, write_parquet
from boti_data.pipelines.sinks_text import CsvSink, CsvSinkConfig, JsonlSink, JsonlSinkConfig

__all__ = [
    "FrameResult",
    "CsvSink",
    "CsvSinkConfig",
    "JsonlSink",
    "JsonlSinkConfig",
    "ParquetDestination",
    "ParquetSink",
    "PipelineSink",
    "SinkWriteResult",
    "awrite_parquet",
    "prepare_partitioned_frame",
    "to_dask_frame",
    "write_parquet",
]
