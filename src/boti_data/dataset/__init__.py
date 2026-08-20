from boti_data._optional import dataframes_required

# Everything re-exported here reaches dask/pandas/polars, so surface the
# missing-extra hint instead of a bare "No module named 'dask'".
with dataframes_required(__name__):
    from boti_data.dataset.hybrid import HybridDataset

__all__ = ["HybridDataset"]
