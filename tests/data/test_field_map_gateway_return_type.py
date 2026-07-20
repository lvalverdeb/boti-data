"""
Laziness guarantees for return_type="auto": eager pandas fetch for small SQL
results, dask fallback for large ones, single-reflection caching, and the
internal request-revalidation bypass on the eager path.

Split out of test_field_map_gateway.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

from typing import Any

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pytest

import boti_data.gateway._series_filters as gateway_series_filters
import boti_data.gateway.return_type as gateway_return_type
import boti_data.gateway.select_cache as gateway_select_cache
from boti_data.gateway import DataFrameParams
from boti_data.gateway.loaders import load_sql, load_sql_partitioned
from boti_data.gateway.requests import SqlLoadRequest

from .conftest import _legacy_gw

# ---------------------------------------------------------------------------
# Laziness guarantees — load() must return dd.DataFrame unless as_pandas=True
# ---------------------------------------------------------------------------


def test_load_returns_lazy_dask_by_default(legacy_dsn) -> None:
    """load() without as_pandas returns a dd.DataFrame, not a pandas DataFrame."""
    gw = _legacy_gw(legacy_dsn)
    try:
        result = gw.load()
        assert isinstance(result, dd.DataFrame), (
            f"Expected dd.DataFrame, got {type(result).__name__}"
        )
    finally:
        gw.close()


def test_load_as_pandas_returns_pandas(legacy_dsn) -> None:
    """load(as_pandas=True) returns a pandas DataFrame."""
    gw = _legacy_gw(legacy_dsn)
    try:
        result = gw.load(as_pandas=True)
        assert isinstance(result, pd.DataFrame), (
            f"Expected pd.DataFrame, got {type(result).__name__}"
        )
    finally:
        gw.close()


def test_configured_gateway_auto_return_type_prefers_pandas_for_small_sql(legacy_dsn) -> None:
    gw = _legacy_gw(legacy_dsn, df_params=DataFrameParams(return_type="auto"))
    try:
        result = gw.load()
        assert isinstance(result, pd.DataFrame)
    finally:
        gw.close()


def test_configured_gateway_auto_uses_eager_fetch_for_small_sql(legacy_dsn, monkeypatch) -> None:
    eager_calls: list[bool] = []
    lazy_calls: list[bool] = []
    real_load_sql = load_sql
    real_load_sql_partitioned = load_sql_partitioned

    def tracking_load_sql(resource, request) -> pd.DataFrame | pa.Table:
        eager_calls.append(True)
        return real_load_sql(resource, request)

    def tracking_load_sql_partitioned(config, resource, request) -> pd.DataFrame | dd.DataFrame:
        lazy_calls.append(True)
        return real_load_sql_partitioned(config, resource, request)

    monkeypatch.setattr("boti_data.gateway._backend_strategies.load_sql", tracking_load_sql)
    monkeypatch.setattr(
        "boti_data.gateway._backend_strategies.load_sql_partitioned", tracking_load_sql_partitioned
    )
    gw = _legacy_gw(legacy_dsn, df_params=DataFrameParams(return_type="auto"))
    try:
        result = gw.load()
        assert isinstance(result, pd.DataFrame)
        assert eager_calls == [True]
        assert not lazy_calls
    finally:
        gw.close()


def test_configured_gateway_reflects_table_once(legacy_dsn, monkeypatch) -> None:
    calls: list[bool] = []
    real_reflect_and_select = gateway_select_cache.reflect_and_select

    def tracking_reflect_and_select(*args, **kwargs) -> tuple[Any, Any]:
        calls.append(True)
        return real_reflect_and_select(*args, **kwargs)

    monkeypatch.setattr(gateway_select_cache, "reflect_and_select", tracking_reflect_and_select)
    gw = _legacy_gw(legacy_dsn)
    try:
        first = gw.load(as_pandas=True)
        second = gw.load(as_pandas=True)
        assert isinstance(first, pd.DataFrame)
        assert isinstance(second, pd.DataFrame)
        assert calls == [True]
    finally:
        gw.close()


def test_configured_gateway_eager_sql_bypasses_internal_request_revalidation(
    legacy_dsn, monkeypatch
) -> None:
    gw = _legacy_gw(
        legacy_dsn,
        df_params=DataFrameParams(return_type="pandas", execution_mode="eager"),
    )
    monkeypatch.setattr(
        SqlLoadRequest,
        "model_validate",
        classmethod(
            lambda cls, *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("model_validate should not run in configured eager SQL path")
            )
        ),
    )
    try:
        result = gw.load()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    finally:
        gw.close()


def test_configured_gateway_auto_return_type_uses_dask_for_large_sql(
    legacy_dsn, monkeypatch
) -> None:
    monkeypatch.setattr(gateway_return_type, "_AUTO_EAGER_MAX_ROWS", 1)
    gw = _legacy_gw(legacy_dsn, df_params=DataFrameParams(return_type="auto"))
    try:
        result = gw.load()
        assert isinstance(result, dd.DataFrame)
    finally:
        gw.close()


@pytest.mark.asyncio
async def test_async_series_filter_resolution_batches_dask_compute(monkeypatch) -> None:
    left = dd.from_pandas(pd.DataFrame({"id": [1, 2, 3]}), npartitions=2)["id"]
    right = dd.from_pandas(pd.DataFrame({"id": [10, 20, 30]}), npartitions=2)["id"]
    calls: list[int] = []
    real_compute = gateway_series_filters.dask.compute

    def tracking_compute(*args, **kwargs) -> tuple[Any, ...]:
        calls.append(len(args))
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(gateway_series_filters.dask, "compute", tracking_compute)

    resolved = await gateway_series_filters.resolve_series_filters_async(
        {"left__in": left, "right__in": right}
    )

    assert resolved["left__in"] == [1, 2, 3]
    assert resolved["right__in"] == [10, 20, 30]
    assert calls == [2]
