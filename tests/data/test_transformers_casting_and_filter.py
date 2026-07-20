"""Tests for boti_data.enrichment.transformers: TypeCaster and RowFilter.

Split out of test_transformers.py purely for god-module headroom.
"""

from __future__ import annotations

import pandas as pd
import pytest

from boti_data.enrichment.transformers import RowFilter, TypeCaster

from ._transformers_shared import _sample_df


@pytest.mark.asyncio
async def test_type_caster_int() -> None:
    df = _sample_df()
    t = TypeCaster({"value": "int64", "active": "bool"})
    result = await t.transform(df)
    assert result["value"].dtype == "int64"
    assert result["active"].dtype == "bool"


@pytest.mark.asyncio
async def test_type_caster_noop_missing_column() -> None:
    df = _sample_df()
    t = TypeCaster({"nonexistent": "int64"})
    result = await t.transform(df)
    pd.testing.assert_frame_equal(result, df)


@pytest.mark.asyncio
async def test_type_caster_partial() -> None:
    df = _sample_df()
    t = TypeCaster({"value": "float64"})
    result = await t.transform(df)
    assert result["value"].dtype == "float64"
    assert result["id"].dtype == "int64"  # unchanged


@pytest.mark.asyncio
async def test_row_filter_basic() -> None:
    df = _sample_df()
    t = RowFilter(["active == 1"])
    result = await t.transform(df)
    assert len(result) == 3
    assert all(result["active"] == 1)


@pytest.mark.asyncio
async def test_row_filter_multiple_predicates() -> None:
    df = _sample_df()
    t = RowFilter(["active == 1", "category == 'x'"])
    result = await t.transform(df)
    # rows with active==1 AND category=='x': rows 0, 2, 4
    assert len(result) == 3
    assert all(result["category"] == "x")


@pytest.mark.asyncio
async def test_row_filter_empty_result() -> None:
    df = _sample_df()
    t = RowFilter(["active == 999"])
    result = await t.transform(df)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_row_filter_rejects_dunder_expression() -> None:
    """Defense against eval sandbox-escape techniques (CVE-2024-9880): dunder
    attribute chains are the building block of every documented escape, so
    predicates containing a dunder-wrapped identifier are refused before
    reaching df.eval()."""
    df = _sample_df()
    t = RowFilter(["active.__class__.__bases__"])
    with pytest.raises(ValueError, match="__"):
        await t.transform(df)


@pytest.mark.asyncio
async def test_row_filter_allows_column_with_embedded_double_underscore() -> None:
    """The guard uses has_dunder_identifier (whole-token match), not a raw
    substring test for "__", so a legitimate column name that merely
    contains "__" in the middle — a realistic dbt/warehouse-style naming
    convention — is not rejected."""
    df = pd.DataFrame({"a__b": [1, 2, 3], "active": [1, 0, 1]})
    t = RowFilter(["a__b > 1"])
    result = await t.transform(df)
    assert list(result["a__b"]) == [2, 3]
