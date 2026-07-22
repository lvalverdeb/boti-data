"""Protocol for DataFrame transformation hooks.

Defines the ``DataFrameTransformer`` contract used by both Silver-layer
transforms (type casting, filtering, derived columns) and Gold-layer
enrichment (lookup merges).  Concrete implementations live in
:mod:`boti_data.enrichment.transformers`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataFrameTransformer(Protocol):
    """Generic DataFrame transform contract.

    Implementations receive a ``pd.DataFrame`` and arbitrary keyword
    arguments (used by higher layers to pass context) and must return
    a ``pd.DataFrame``.

    This protocol is intentionally minimal — ``**kwargs`` keeps it
    flexible so that etl-core can pass a ``TransformContext`` without
    boti-data knowing about it.
    """

    async def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame: ...
