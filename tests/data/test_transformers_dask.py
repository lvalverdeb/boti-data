"""Dask-frame regression tests for boti_data.enrichment.transformers.

Split out from test_transformers_casting_and_filter.py because these need a
real dd.DataFrame fixture rather than the shared pandas _sample_df().
"""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import pytest

from boti_data.enrichment.transformers import TypeCaster

from ._transformers_shared import _sample_df


def _sample_dd(npartitions: int = 2) -> dd.DataFrame:
    return dd.from_pandas(_sample_df(), npartitions=npartitions)


@pytest.mark.asyncio
async def test_type_caster_casts_dask_frame() -> None:
    """dd.DataFrame.astype() has no errors= kwarg; TypeCaster must branch
    on frame type rather than always passing errors="ignore" (which raised
    TypeError unconditionally against any dask input before this fix)."""
    ddf = _sample_dd()
    t = TypeCaster({"value": "int64", "active": "bool"})
    result = await t.transform(ddf)
    assert isinstance(result, dd.DataFrame)
    computed = result.compute()
    assert computed["value"].dtype == "int64"
    assert computed["active"].dtype == "bool"


@pytest.mark.asyncio
async def test_type_caster_noop_missing_column_dask() -> None:
    ddf = _sample_dd()
    t = TypeCaster({"nonexistent": "int64"})
    result = await t.transform(ddf)
    pd.testing.assert_frame_equal(result.compute(), ddf.compute())


@pytest.mark.asyncio
async def test_type_caster_dask_raises_instead_of_ignoring_failed_cast() -> None:
    """Unlike the pandas path (errors="ignore" silently skips a column that
    can't cast), dask has no partial-success concept for a single column's
    dtype — a genuinely uncastable column raises rather than being skipped.
    This is a documented semantic difference, not a bug: a dask column's
    dtype must be uniform across every partition, so there is no meaningful
    "ignore" fallback to offer."""
    ddf = dd.from_pandas(pd.DataFrame({"value": ["not-a-number", "also-not"]}), npartitions=1)
    t = TypeCaster({"value": "int64"})
    result = await t.transform(ddf)
    with pytest.raises((ValueError, TypeError)):
        result.compute()
