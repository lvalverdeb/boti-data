from __future__ import annotations

from boti_data._optional import dataframes_required

# Everything re-exported here reaches dask/pandas/polars, so surface the
# missing-extra hint instead of a bare "No module named 'dask'".
with dataframes_required(__name__):
    from boti_data.enrichment.async_enricher import AsyncFrameEnricher, FrameEnricher
    from boti_data.enrichment.composite import CompositeTransformer
    from boti_data.enrichment.protocol import DataFrameTransformer
    from boti_data.enrichment.specs import AttachmentSpec
    from boti_data.enrichment.transformers import (
        Deduplicator,
        DerivedColumn,
        RowFilter,
        TypeCaster,
    )

__all__ = [
    "AttachmentSpec",
    "AsyncFrameEnricher",
    "CompositeTransformer",
    "Deduplicator",
    "DataFrameTransformer",
    "DerivedColumn",
    "FrameEnricher",
    "RowFilter",
    "TypeCaster",
]
