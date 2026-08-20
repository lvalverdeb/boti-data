from boti_data._optional import dataframes_required

# Everything re-exported here reaches dask/pandas/polars, so surface the
# missing-extra hint instead of a bare "No module named 'dask'".
with dataframes_required(__name__):
    from boti_data.watermark.incremental import (
        IncrementalResult,
        advance_watermark,
        build_incremental_filters,
    )
    from boti_data.watermark.store import FileWatermarkStore, FsspecWatermarkStore, WatermarkStore

__all__ = [
    "FileWatermarkStore",
    "FsspecWatermarkStore",
    "IncrementalResult",
    "WatermarkStore",
    "advance_watermark",
    "build_incremental_filters",
]
