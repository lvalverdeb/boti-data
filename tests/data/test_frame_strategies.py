"""
Unit tests for frame_strategies.py – covers normalize, concat, rename, select,
apply_column_names, has_any_rows, apply_options, and get_frame_strategy for all
four engine strategies (pandas, dask, arrow, polars).
"""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from boti_data.gateway.frame_strategies import (
    ArrowFrameStrategy,
    DaskFrameStrategy,
    PandasFrameStrategy,
    PolarsFrameStrategy,
    get_frame_strategy,
)
from boti_data.gateway.requests import DataFrameOptions, DataFrameParams

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_pd() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture()
def empty_pd() -> pd.DataFrame:
    return pd.DataFrame({"a": [], "b": []})


@pytest.fixture()
def simple_dask(simple_pd) -> dd.DataFrame:
    return dd.from_pandas(simple_pd, npartitions=1)


@pytest.fixture()
def simple_arrow(simple_pd) -> pa.Table:
    return pa.Table.from_pandas(simple_pd, preserve_index=False)


@pytest.fixture()
def simple_polars(simple_pd) -> pl.DataFrame:
    return pl.from_pandas(simple_pd)


@pytest.fixture()
def no_opts() -> DataFrameOptions:
    return DataFrameOptions()


@pytest.fixture()
def no_params() -> DataFrameParams:
    return DataFrameParams()


# ---------------------------------------------------------------------------
# get_frame_strategy
# ---------------------------------------------------------------------------


def test_get_frame_strategy_returns_correct_types() -> None:
    assert isinstance(get_frame_strategy("pandas"), PandasFrameStrategy)
    assert isinstance(get_frame_strategy("dask"), DaskFrameStrategy)
    assert isinstance(get_frame_strategy("arrow"), ArrowFrameStrategy)
    assert isinstance(get_frame_strategy("polars"), PolarsFrameStrategy)


def test_get_frame_strategy_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported frame engine"):
        get_frame_strategy("unknown")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PandasFrameStrategy
# ---------------------------------------------------------------------------


class TestPandasFrameStrategy:
    s = PandasFrameStrategy()

    def test_normalize_returns_dataframe(self, simple_pd) -> None:
        result = self.s.normalize(simple_pd)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]

    def test_normalize_from_arrow(self, simple_arrow) -> None:
        result = self.s.normalize(simple_arrow)
        assert isinstance(result, pd.DataFrame)

    def test_normalize_from_polars(self, simple_polars) -> None:
        result = self.s.normalize(simple_polars)
        assert isinstance(result, pd.DataFrame)

    def test_has_any_rows_true(self, simple_pd) -> None:
        assert self.s.has_any_rows(simple_pd) is True

    def test_has_any_rows_false(self, empty_pd) -> None:
        assert self.s.has_any_rows(empty_pd) is False

    def test_concat(self, simple_pd) -> None:
        result = self.s.concat([simple_pd, simple_pd])
        assert len(result) == 6

    def test_rename_columns(self, simple_pd) -> None:
        result = self.s.rename_columns(simple_pd, {"a": "alpha"})
        assert "alpha" in result.columns
        assert "a" not in result.columns

    def test_rename_columns_empty_map(self, simple_pd) -> None:
        result = self.s.rename_columns(simple_pd, {})
        assert list(result.columns) == ["a", "b"]

    def test_select_columns(self, simple_pd) -> None:
        result = self.s.select_columns(simple_pd, ["a"])
        assert list(result.columns) == ["a"]

    def test_apply_column_names(self, simple_pd) -> None:
        result = self.s.apply_column_names(simple_pd, ["col1", "col2"])
        assert list(result.columns) == ["col1", "col2"]

    def test_apply_column_names_wrong_length(self, simple_pd) -> None:
        with pytest.raises(ValueError, match="column_names length"):
            self.s.apply_column_names(simple_pd, ["only_one"])

    def test_apply_options_sort(self, simple_pd, no_params) -> None:
        opts = DataFrameOptions(sort_field="a")
        result = self.s.apply_options(simple_pd.iloc[::-1].reset_index(drop=True), no_params, opts)
        assert list(result["a"]) == sorted(result["a"])

    def test_apply_options_dedup(self, no_params) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        opts = DataFrameOptions(duplicate_expr="a", duplicate_keep="first")
        result = self.s.apply_options(df, no_params, opts)
        assert len(result) == 2

    def test_apply_options_noop(self, simple_pd, no_params, no_opts) -> None:
        result = self.s.apply_options(simple_pd, no_params, no_opts)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# DaskFrameStrategy
# ---------------------------------------------------------------------------


class TestDaskFrameStrategy:
    s = DaskFrameStrategy()

    def test_normalize_returns_dask(self, simple_pd) -> None:
        result = self.s.normalize(simple_pd)
        assert isinstance(result, dd.DataFrame)

    def test_normalize_dask_passthrough(self, simple_dask) -> None:
        result = self.s.normalize(simple_dask)
        assert result is simple_dask

    def test_has_any_rows_true(self, simple_dask) -> None:
        assert self.s.has_any_rows(simple_dask) is True

    def test_concat(self, simple_dask) -> None:
        result = self.s.concat([simple_dask, simple_dask])
        assert isinstance(result, dd.DataFrame)
        assert len(result.compute()) == 6

    def test_rename_columns(self, simple_dask) -> None:
        result = self.s.rename_columns(simple_dask, {"a": "alpha"})
        assert "alpha" in result.columns

    def test_select_columns(self, simple_dask) -> None:
        result = self.s.select_columns(simple_dask, ["a"])
        assert list(result.columns) == ["a"]

    def test_apply_column_names(self, simple_dask) -> None:
        result = self.s.apply_column_names(simple_dask, ["col1", "col2"])
        assert list(result.columns) == ["col1", "col2"]

    def test_persist_returns_dask(self, simple_dask) -> None:
        result = self.s.persist(simple_dask)
        assert isinstance(result, dd.DataFrame)


# ---------------------------------------------------------------------------
# ArrowFrameStrategy
# ---------------------------------------------------------------------------


class TestArrowFrameStrategy:
    s = ArrowFrameStrategy()

    def test_normalize_returns_table(self, simple_arrow) -> None:
        result = self.s.normalize(simple_arrow)
        assert isinstance(result, pa.Table)

    def test_normalize_from_pandas(self, simple_pd) -> None:
        result = self.s.normalize(simple_pd)
        assert isinstance(result, pa.Table)

    def test_has_any_rows_true(self, simple_arrow) -> None:
        assert self.s.has_any_rows(simple_arrow) is True

    def test_has_any_rows_false(self) -> None:
        empty = pa.table({"a": pa.array([], type=pa.int64())})
        assert self.s.has_any_rows(empty) is False

    def test_concat(self, simple_arrow) -> None:
        result = self.s.concat([simple_arrow, simple_arrow])
        assert result.num_rows == 6

    def test_rename_columns(self, simple_arrow) -> None:
        result = self.s.rename_columns(simple_arrow, {"a": "alpha"})
        assert "alpha" in result.column_names

    def test_rename_columns_empty_map(self, simple_arrow) -> None:
        result = self.s.rename_columns(simple_arrow, {})
        assert result is simple_arrow

    def test_select_columns(self, simple_arrow) -> None:
        result = self.s.select_columns(simple_arrow, ["a"])
        assert result.column_names == ["a"]

    def test_apply_column_names(self, simple_arrow) -> None:
        result = self.s.apply_column_names(simple_arrow, ["col1", "col2"])
        assert result.column_names == ["col1", "col2"]

    def test_apply_options_raises_on_index_col(self, simple_arrow) -> None:
        opts = DataFrameOptions()
        params = DataFrameParams(index_col="a")
        with pytest.raises(ValueError, match="Arrow return_type does not support"):
            self.s.apply_options(simple_arrow, params, opts)


# ---------------------------------------------------------------------------
# PolarsFrameStrategy
# ---------------------------------------------------------------------------


class TestPolarsFrameStrategy:
    s = PolarsFrameStrategy()

    def test_normalize_returns_polars(self, simple_polars) -> None:
        result = self.s.normalize(simple_polars)
        assert isinstance(result, pl.DataFrame)

    def test_normalize_from_pandas(self, simple_pd) -> None:
        result = self.s.normalize(simple_pd)
        assert isinstance(result, pl.DataFrame)

    def test_has_any_rows_true(self, simple_polars) -> None:
        assert self.s.has_any_rows(simple_polars) is True

    def test_has_any_rows_false(self) -> None:
        empty = pl.DataFrame({"a": [], "b": []})
        assert self.s.has_any_rows(empty) is False

    def test_concat(self, simple_polars) -> None:
        result = self.s.concat([simple_polars, simple_polars])
        assert result.height == 6

    def test_rename_columns(self, simple_polars) -> None:
        result = self.s.rename_columns(simple_polars, {"a": "alpha"})
        assert "alpha" in result.columns

    def test_select_columns(self, simple_polars) -> None:
        result = self.s.select_columns(simple_polars, ["a"])
        assert result.columns == ["a"]

    def test_apply_column_names(self, simple_polars) -> None:
        result = self.s.apply_column_names(simple_polars, ["col1", "col2"])
        assert result.columns == ["col1", "col2"]

    def test_apply_options_sort(self, simple_polars, no_params) -> None:
        opts = DataFrameOptions(sort_field="a")
        df = simple_polars.sort("a", descending=True)
        result = self.s.apply_options(df, no_params, opts)
        assert result["a"].to_list() == sorted(result["a"].to_list())

    def test_apply_options_dedup(self, no_params) -> None:
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        opts = DataFrameOptions(duplicate_expr="a", duplicate_keep="first")
        result = self.s.apply_options(df, no_params, opts)
        assert result.height == 2

    def test_apply_options_raises_on_datetime_index(self, simple_polars) -> None:
        opts = DataFrameOptions()
        params = DataFrameParams(datetime_index="a")
        with pytest.raises(ValueError, match="Polars return_type does not support"):
            self.s.apply_options(simple_polars, params, opts)
