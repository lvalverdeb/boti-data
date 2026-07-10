"""Tests for boti_data.enrichment.transformers and CompositeTransformer."""

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

# ── helpers ──────────────────────────────────────────────────────────────────


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["a", "b", "c", "d", "e"],
        "value": ["10", "20", "30", "40", "50"],
        "active": [1, 0, 1, 0, 1],
        "category": ["x", "y", "x", "y", "x"],
    })


# ── TypeCaster ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_type_caster_int():
    df = _sample_df()
    t = TypeCaster({"value": "int64", "active": "bool"})
    result = await t.transform(df)
    assert result["value"].dtype == "int64"
    assert result["active"].dtype == "bool"


@pytest.mark.asyncio
async def test_type_caster_noop_missing_column():
    df = _sample_df()
    t = TypeCaster({"nonexistent": "int64"})
    result = await t.transform(df)
    pd.testing.assert_frame_equal(result, df)


@pytest.mark.asyncio
async def test_type_caster_partial():
    df = _sample_df()
    t = TypeCaster({"value": "float64"})
    result = await t.transform(df)
    assert result["value"].dtype == "float64"
    assert result["id"].dtype == "int64"  # unchanged


# ── RowFilter ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_row_filter_basic():
    df = _sample_df()
    t = RowFilter(["active == 1"])
    result = await t.transform(df)
    assert len(result) == 3
    assert all(result["active"] == 1)


@pytest.mark.asyncio
async def test_row_filter_multiple_predicates():
    df = _sample_df()
    t = RowFilter(["active == 1", "category == 'x'"])
    result = await t.transform(df)
    # rows with active==1 AND category=='x': rows 0, 2, 4
    assert len(result) == 3
    assert all(result["category"] == "x")


@pytest.mark.asyncio
async def test_row_filter_empty_result():
    df = _sample_df()
    t = RowFilter(["active == 999"])
    result = await t.transform(df)
    assert len(result) == 0


# ── DerivedColumn ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_derived_column_simple():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    t = DerivedColumn({"doubled": "a * 2"})
    result = await t.transform(df)
    assert "doubled" in result.columns
    assert list(result["doubled"]) == [2, 4, 6]


@pytest.mark.asyncio
async def test_derived_column_multiple():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    t = DerivedColumn({"sum": "a + b", "product": "a * b"})
    result = await t.transform(df)
    assert list(result["sum"]) == [5, 7, 9]
    assert list(result["product"]) == [4, 10, 18]


# ── Deduplicator ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deduplicator_basic():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
    t = Deduplicator(subset=["a"], keep="first")
    result = await t.transform(df)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_deduplicator_no_duplicates():
    df = _sample_df()
    t = Deduplicator()
    result = await t.transform(df)
    assert len(result) == 5


# ── CompositeTransformer ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composite_chain():
    df = pd.DataFrame({"a": ["1", "2", "3"], "b": [10, 20, 30]})
    chain = CompositeTransformer([
        TypeCaster({"a": "int64"}),
        DerivedColumn({"total": "a + b"}),
    ])
    result = await chain.transform(df)
    assert result["a"].dtype == "int64"
    assert "total" in result.columns


@pytest.mark.asyncio
async def test_composite_empty():
    df = _sample_df()
    chain = CompositeTransformer([])
    result = await chain.transform(df)
    pd.testing.assert_frame_equal(result, df)


@pytest.mark.asyncio
async def test_composite_stops_on_error():
    class BadTransformer:
        async def transform(self, df, **kwargs):
            raise ValueError("boom")

    chain = CompositeTransformer([TypeCaster({"x": "int64"}), BadTransformer()])
    with pytest.raises(ValueError, match="boom"):
        await chain.transform(_sample_df())


@pytest.mark.asyncio
async def test_composite_len_repr():
    chain = CompositeTransformer([TypeCaster({}), RowFilter([])])
    assert len(chain) == 2
    assert "TypeCaster" in repr(chain)
    assert "RowFilter" in repr(chain)


# ── Protocol conformance ─────────────────────────────────────────────────────


def test_transformers_satisfy_protocol():
    for cls in (TypeCaster, RowFilter, DerivedColumn, Deduplicator):
        assert issubclass(cls, DataFrameTransformer)
