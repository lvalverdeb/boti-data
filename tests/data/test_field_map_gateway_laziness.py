"""
Lazy dd.DataFrame result semantics: correct columns/fieldnames/sticky-filters
under laziness, compute() parity with as_pandas=True, aload() laziness, and
semi_join() laziness (including the series-list-resolution short-circuit).

Split out of test_field_map_gateway.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

from typing import Any

import dask.dataframe as dd
import pandas as pd
import pytest

import boti_data.gateway._series_filters as gateway_series_filters
from boti_data.gateway import DataFrameParams

from .conftest import _legacy_gw


def test_lazy_result_has_correct_columns(legacy_dsn) -> None:
    """Lazy dd.DataFrame must carry semantic column names (field_map applied)."""
    gw = _legacy_gw(legacy_dsn)
    try:
        result = gw.load()
        assert isinstance(result, dd.DataFrame)
        assert "product_type_id" in result.columns
        assert "id_tipo_produto" not in result.columns
    finally:
        gw.close()


def test_lazy_result_fieldnames_pipeline(legacy_dsn) -> None:
    """fieldnames filter preserves laziness — result is still a dd.DataFrame."""
    gw = _legacy_gw(
        legacy_dsn,
        df_params=DataFrameParams(fieldnames=("global_track_id", "barcode")),
    )
    try:
        result = gw.load()
        assert isinstance(result, dd.DataFrame)
        assert set(result.columns) == {"global_track_id", "barcode"}
    finally:
        gw.close()


def test_lazy_result_sticky_filter_preserves_dask(legacy_dsn) -> None:
    """Sticky filters must not force an eager compute."""
    gw = _legacy_gw(legacy_dsn, sticky_filters={"product_type_id": 1})
    try:
        result = gw.load()
        assert isinstance(result, dd.DataFrame)
        # Compute only here — result must materialise correctly
        pdf = result.compute()
        assert len(pdf) == 2
        assert set(pdf["product_type_id"].tolist()) == {1}
    finally:
        gw.close()


def test_lazy_result_computes_correct_data(legacy_dsn) -> None:
    """Lazy result must compute to the same data as as_pandas=True."""
    gw = _legacy_gw(legacy_dsn)
    try:
        lazy = gw.load()
        assert isinstance(lazy, dd.DataFrame)
        eager = gw.load(as_pandas=True)
        assert isinstance(eager, pd.DataFrame)
        pd.testing.assert_frame_equal(
            lazy.compute().sort_values("global_track_id").reset_index(drop=True),
            eager.sort_values("global_track_id").reset_index(drop=True),
        )
    finally:
        gw.close()


@pytest.mark.asyncio
async def test_aload_returns_lazy_dask_by_default(legacy_dsn) -> None:
    """aload() without as_pandas returns a dd.DataFrame."""
    gw = _legacy_gw(legacy_dsn)
    try:
        result = await gw.aload()
        assert isinstance(result, dd.DataFrame), (
            f"Expected dd.DataFrame, got {type(result).__name__}"
        )
    finally:
        gw.close()


@pytest.mark.asyncio
async def test_aload_as_pandas_returns_pandas(legacy_dsn) -> None:
    """aload(as_pandas=True) returns a pandas DataFrame."""
    gw = _legacy_gw(legacy_dsn)
    try:
        result = await gw.aload(as_pandas=True)
        assert isinstance(result, pd.DataFrame)
    finally:
        gw.close()


def test_semi_join_without_as_pandas_returns_dask(legacy_dsn) -> None:
    """semi_join() default result is a lazy dd.DataFrame."""
    gw = _legacy_gw(legacy_dsn)
    try:
        result = gw.semi_join(pd.Series([10, 30]), on="global_track_id")
        assert isinstance(result, dd.DataFrame)
        pdf = result.compute()
        assert set(pdf["global_track_id"].tolist()) == {10, 30}
    finally:
        gw.close()


def test_lazy_semi_join_avoids_series_list_resolution(legacy_dsn, monkeypatch) -> None:
    gw = _legacy_gw(legacy_dsn)

    def forbid_series_resolution(options) -> dict[str, Any]:
        if any(isinstance(value, (pd.Series, dd.Series)) for value in options.values()):
            raise AssertionError(
                "series should not reach generic filter resolution in lazy semi_join"
            )
        return options

    monkeypatch.setattr(gateway_series_filters, "resolve_series_filters", forbid_series_resolution)
    try:
        result = gw.semi_join(pd.Series([10, 30]), on="global_track_id")
        assert isinstance(result, dd.DataFrame)
        assert set(result.compute()["global_track_id"].tolist()) == {10, 30}
    finally:
        gw.close()
