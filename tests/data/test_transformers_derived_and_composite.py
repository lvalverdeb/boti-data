"""Tests for boti_data.enrichment.transformers: DerivedColumn, Deduplicator,
CompositeTransformer, and DataFrameTransformer protocol conformance.

Split out of test_transformers.py purely for god-module headroom.
"""

from __future__ import annotations

import pandas as pd
import pytest

from boti_data.enrichment.composite import CompositeTransformer
from boti_data.enrichment.protocol import DataFrameTransformer
from boti_data.enrichment.transformers import (
    Deduplicator,
    DerivedColumn,
    RowFilter,
    TypeCaster,
)

from ._transformers_shared import _sample_df


@pytest.mark.asyncio
async def test_derived_column_simple() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    t = DerivedColumn({"doubled": "a * 2"})
    result = await t.transform(df)
    assert "doubled" in result.columns
    assert list(result["doubled"]) == [2, 4, 6]


@pytest.mark.asyncio
async def test_derived_column_multiple() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    t = DerivedColumn({"sum": "a + b", "product": "a * b"})
    result = await t.transform(df)
    assert list(result["sum"]) == [5, 7, 9]
    assert list(result["product"]) == [4, 10, 18]


@pytest.mark.asyncio
async def test_derived_column_rejects_dunder_expression() -> None:
    """Defense against eval sandbox-escape techniques (CVE-2024-9880): dunder
    attribute chains are the building block of every documented escape, so
    expressions containing a dunder-wrapped identifier are refused before
    reaching df.eval()."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    t = DerivedColumn({"pwned": "a.__class__.__mro__"})
    with pytest.raises(ValueError, match="__"):
        await t.transform(df)


@pytest.mark.asyncio
async def test_derived_column_allows_column_with_embedded_double_underscore() -> None:
    """Same has_dunder_identifier whole-token guard as RowFilter — a column
    referencing "a__b" is not a dunder attribute chain and must not be
    rejected."""
    df = pd.DataFrame({"a__b": [1, 2, 3]})
    t = DerivedColumn({"doubled": "a__b * 2"})
    result = await t.transform(df)
    assert list(result["doubled"]) == [2, 4, 6]


@pytest.mark.asyncio
async def test_deduplicator_basic() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
    t = Deduplicator(subset=["a"], keep="first")
    result = await t.transform(df)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_deduplicator_no_duplicates() -> None:
    df = _sample_df()
    t = Deduplicator()
    result = await t.transform(df)
    assert len(result) == 5


@pytest.mark.asyncio
async def test_composite_chain() -> None:
    df = pd.DataFrame({"a": ["1", "2", "3"], "b": [10, 20, 30]})
    chain = CompositeTransformer(
        [
            TypeCaster({"a": "int64"}),
            DerivedColumn({"total": "a + b"}),
        ]
    )
    result = await chain.transform(df)
    assert result["a"].dtype == "int64"
    assert "total" in result.columns


@pytest.mark.asyncio
async def test_composite_empty() -> None:
    df = _sample_df()
    chain = CompositeTransformer([])
    result = await chain.transform(df)
    pd.testing.assert_frame_equal(result, df)


@pytest.mark.asyncio
async def test_composite_stops_on_error() -> None:
    class BadTransformer:
        async def transform(self, df, **kwargs) -> None:
            raise ValueError("boom")

    chain = CompositeTransformer([TypeCaster({"x": "int64"}), BadTransformer()])
    with pytest.raises(ValueError, match="boom"):
        await chain.transform(_sample_df())


@pytest.mark.asyncio
async def test_composite_len_repr() -> None:
    chain = CompositeTransformer([TypeCaster({}), RowFilter([])])
    assert len(chain) == 2
    assert "TypeCaster" in repr(chain)
    assert "RowFilter" in repr(chain)


def test_transformers_satisfy_protocol() -> None:
    for cls in (TypeCaster, RowFilter, DerivedColumn, Deduplicator):
        assert issubclass(cls, DataFrameTransformer)
