"""Distributed semi-join support for DataGateway.semi_join()/asemi_join()."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import dask.dataframe as dd
import pandas as pd
import polars as pl

from . import _series_filters
from .frame_strategies import FrameResult
from .requests import DataFrameParams


class SemiJoinService:
    """Decides whether a semi-join can be resolved lazily, and performs it.

    Split out of DataGateway: heavily coupled to gateway state, but neither
    method is referenced directly by tests, so both move here wholesale
    rather than staying as thin wrappers.
    """

    def __init__(
        self,
        *,
        configured: bool,
        df_params: DataFrameParams,
        default_return_type: str,
        load: Callable[..., FrameResult],
    ) -> None:
        self._configured = configured
        self._df_params = df_params
        self._default_return_type = default_return_type
        self._load = load

    def _passes_lazy_semi_join_guards(self, options: dict[str, Any]) -> bool:
        """Cheap option checks that must all hold before a lazy semi-join is considered."""
        if (
            options.get("as_pandas")
            or options.get("execution_mode") == "eager"
            or options.get("partitioned") is False
        ):
            return False
        requested_return_type = options.get(
            "return_type",
            self._default_return_type if self._configured else "dask",
        )
        return requested_return_type == "dask"

    @staticmethod
    def _structured_selects_column(on: str, options: dict[str, Any]) -> bool:
        statement = options.get("statement")
        if statement is None:
            return False
        selected_names = [
            str(getattr(selected, "key", None) or getattr(selected, "name", None))
            for selected in statement.selected_columns
        ]
        return on in selected_names

    def supports_lazy_series_semi_join(
        self,
        join_series: pd.Series | dd.Series | pl.Series,
        on: str,
        options: dict[str, Any],
    ) -> bool:
        if not self._passes_lazy_semi_join_guards(options):
            return False
        if self._configured:
            return not self._df_params.fieldnames or on in self._df_params.fieldnames
        return self._structured_selects_column(on, options)

    def lazy_series_semi_join(
        self,
        join_series: pd.Series | dd.Series | pl.Series,
        *,
        on: str,
        options: dict[str, Any],
    ) -> dd.DataFrame:
        base_options = {
            key: value
            for key, value in options.items()
            if key not in {"return_type", "execution_mode", "as_pandas", f"{on}__in"}
        }
        frame = self._load(
            **base_options,
            return_type="dask",
            execution_mode="lazy",
        )
        assert isinstance(frame, dd.DataFrame)
        key_frame = _series_filters.series_to_dask_key_frame(join_series, column_name=on)
        if on in frame.columns:
            key_frame = key_frame.astype({on: frame.dtypes[on]})
        joined = frame.merge(key_frame, how="inner", on=on)
        return joined[list(frame.columns)]
