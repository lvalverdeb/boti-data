"""Composition primitive for chaining DataFrame transformers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from boti_data.enrichment.protocol import DataFrameTransformer


class CompositeTransformer:
    """Chains multiple :class:`DataFrameTransformer` instances in sequence.

    The output of each transformer is fed as input to the next.
    If any transformer raises, the chain stops and the exception propagates.

    Parameters
    ----------
    transformers:
        Ordered sequence of transformers to apply.
    """

    def __init__(self, transformers: Sequence[DataFrameTransformer]) -> None:
        self._transformers = list(transformers)

    async def transform(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        for transformer in self._transformers:
            df = await transformer.transform(df, **kwargs)
        return df

    def __len__(self) -> int:
        return len(self._transformers)

    def __repr__(self) -> str:
        names = [type(t).__name__ for t in self._transformers]
        return f"CompositeTransformer({', '.join(names)})"
