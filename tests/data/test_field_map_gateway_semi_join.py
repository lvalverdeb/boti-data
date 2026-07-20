"""
Semi-join and Series-as-__in-filter behavior: pd.Series/dd.Series resolution,
semi_join()/asemi_join() sugar, and resolve_series_filters() passthrough.

Split out of test_field_map_gateway.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import pandas as pd
import pytest

import boti_data.gateway._series_filters as gateway_series_filters

from .conftest import _legacy_gw

# ---------------------------------------------------------------------------
# Semi-join: Series as __in filter value (item 6)
# ---------------------------------------------------------------------------


def test_load_pandas_series_as_in_filter(legacy_dsn) -> None:
    """pd.Series passed as field__in resolves to unique-value list before query."""
    gw = _legacy_gw(legacy_dsn)
    try:
        series = pd.Series([10, 20, 10])  # duplicate 10 should be deduped
        df = gw.load(global_track_id__in=series, as_pandas=True)
        assert len(df) == 2
        assert set(df["global_track_id"].tolist()) == {10, 20}
    finally:
        gw.close()


def test_load_pandas_series_with_nan_drops_nulls(legacy_dsn) -> None:
    """NaN values in a pd.Series __in filter are silently discarded."""

    gw = _legacy_gw(legacy_dsn)
    try:
        series = pd.Series([10.0, float("nan")])
        df = gw.load(global_track_id__in=series, as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["global_track_id"] == 10
    finally:
        gw.close()


def test_load_dask_series_as_in_filter(legacy_dsn) -> None:
    """dd.Series passed as field__in is computed and resolved before query."""
    import dask.dataframe as dd

    gw = _legacy_gw(legacy_dsn)
    try:
        pdf = pd.DataFrame({"id": [10, 30, 30]})
        ddf = dd.from_pandas(pdf, npartitions=2)
        df = gw.load(global_track_id__in=ddf["id"], as_pandas=True)
        assert len(df) == 2
        assert set(df["global_track_id"].tolist()) == {10, 30}
    finally:
        gw.close()


@pytest.mark.asyncio
async def test_aload_pandas_series_as_in_filter(legacy_dsn) -> None:
    """Async path: pd.Series __in resolves correctly."""
    gw = _legacy_gw(legacy_dsn)
    try:
        series = pd.Series([20, 30])
        df = await gw.aload(global_track_id__in=series, as_pandas=True)
        assert len(df) == 2
        assert set(df["global_track_id"].tolist()) == {20, 30}
    finally:
        gw.close()


@pytest.mark.asyncio
async def test_aload_dask_series_as_in_filter(legacy_dsn) -> None:
    """Async path: dd.Series is computed in a thread pool before query."""
    import dask.dataframe as dd

    gw = _legacy_gw(legacy_dsn)
    try:
        pdf = pd.DataFrame({"gid": [10, 20]})
        ddf = dd.from_pandas(pdf, npartitions=1)
        df = await gw.aload(global_track_id__in=ddf["gid"], as_pandas=True)
        assert len(df) == 2
        assert set(df["global_track_id"].tolist()) == {10, 20}
    finally:
        gw.close()


def test_semi_join_method(legacy_dsn) -> None:
    """semi_join() is sugar for load(field__in=series)."""
    gw = _legacy_gw(legacy_dsn)
    try:
        series = pd.Series([10, 30])
        df = gw.semi_join(series, on="global_track_id", as_pandas=True)
        assert len(df) == 2
        assert set(df["global_track_id"].tolist()) == {10, 30}
    finally:
        gw.close()


@pytest.mark.asyncio
async def test_asemi_join_method(legacy_dsn) -> None:
    """asemi_join() is sugar for aload(field__in=series)."""
    gw = _legacy_gw(legacy_dsn)
    try:
        series = pd.Series([20])
        df = await gw.asemi_join(series, on="global_track_id", as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["global_track_id"] == 20
    finally:
        gw.close()


def test_semi_join_with_field_map_translates_column(legacy_dsn) -> None:
    """semi_join uses semantic names; field_map handles DB translation."""
    gw = _legacy_gw(legacy_dsn, sticky_filters={"product_type_id": 1})
    try:
        # Both sticky filter (type=1) AND semi-join must apply
        series = pd.Series([10])
        df = gw.semi_join(series, on="global_track_id", as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["product_type_id"] == 1
        assert df.iloc[0]["global_track_id"] == 10
        assert "id_track_global" not in df.columns
    finally:
        gw.close()


def test_resolve_series_filters_is_noop_without_series() -> None:
    """resolve_series_filters returns the same dict when no Series are present."""
    opts = {"field__in": [1, 2, 3], "limit": 10}
    result = gateway_series_filters.resolve_series_filters(opts)
    assert result is opts  # same object, no copy made
