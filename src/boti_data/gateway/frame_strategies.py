from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Literal

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa

from boti_data.db.arrow_schema_mapper import arrow_table_to_pandas

from .arrow_adapters import apply_table_options
from .requests import DataFrameOptions, DataFrameParams, ResolvedReturnType

_log = logging.getLogger(__name__)

LoaderReturnType = Literal["pandas", "arrow", "dask"]
FrameResult = pd.DataFrame | dd.DataFrame | pa.Table | pl.DataFrame
FrameSeries = pd.Series | dd.Series | pl.Series


def _to_pandas(frame: FrameResult) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    if isinstance(frame, dd.DataFrame):
        return frame.compute()
    if isinstance(frame, pa.Table):
        return arrow_table_to_pandas(frame)
    if isinstance(frame, pl.DataFrame):
        return frame.to_pandas()
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _to_arrow(frame: FrameResult) -> pa.Table:
    if isinstance(frame, pa.Table):
        return frame
    if isinstance(frame, dd.DataFrame):
        frame = frame.compute()
    if isinstance(frame, pd.DataFrame):
        return pa.Table.from_pandas(frame, preserve_index=False)
    if isinstance(frame, pl.DataFrame):
        return frame.to_arrow()
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _to_polars(frame: FrameResult) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame
    return pl.from_arrow(_to_arrow(frame))


def _normalize_duplicate_keep(value: str | bool) -> str:
    return "none" if value is False else value


def _build_polars_group_exprs(
    dataframe: pl.DataFrame,
    group_keys: list[str],
    group_expr: str | dict[str, str],
) -> list[pl.Expr]:
    if isinstance(group_expr, str):
        aggregations = {
            name: group_expr for name in dataframe.columns if name not in set(group_keys)
        }
    else:
        aggregations = dict(group_expr)

    exprs: list[pl.Expr] = []
    for column, func in aggregations.items():
        base = pl.col(column)
        if func == "first":
            exprs.append(base.first().alias(column))
        elif func == "last":
            exprs.append(base.last().alias(column))
        elif func == "sum":
            exprs.append(base.sum().alias(column))
        elif func == "min":
            exprs.append(base.min().alias(column))
        elif func == "max":
            exprs.append(base.max().alias(column))
        elif func == "mean":
            exprs.append(base.mean().alias(column))
        elif func in {"count", "len"}:
            exprs.append(base.count().alias(column))
        elif func == "any":
            exprs.append(base.any().alias(column))
        else:
            raise ValueError(f"Unsupported polars aggregation: {func!r}")
    return exprs


class FrameStrategy(ABC):
    engine: ResolvedReturnType
    loader_return_type: LoaderReturnType

    @abstractmethod
    def normalize(self, frame: FrameResult) -> FrameResult:
        raise NotImplementedError

    @abstractmethod
    def concat(self, frames: list[FrameResult]) -> FrameResult:
        raise NotImplementedError

    @abstractmethod
    def has_any_rows(self, frame: FrameResult) -> bool:
        raise NotImplementedError

    @abstractmethod
    def rename_columns(self, frame: FrameResult, rename_map: dict[str, str]) -> FrameResult:
        raise NotImplementedError

    @abstractmethod
    def select_columns(self, frame: FrameResult, columns: list[str]) -> FrameResult:
        raise NotImplementedError

    @abstractmethod
    def apply_column_names(self, frame: FrameResult, column_names: list[str]) -> FrameResult:
        raise NotImplementedError

    @abstractmethod
    def apply_options(
        self,
        frame: FrameResult,
        params: DataFrameParams,
        opts: DataFrameOptions,
    ) -> FrameResult:
        raise NotImplementedError

    def persist(self, frame: FrameResult) -> FrameResult:
        return frame

    def resolve_series(self, value: Any) -> Any:
        if isinstance(value, dd.Series):
            return value.compute().dropna().unique().tolist()
        if isinstance(value, pd.Series):
            return value.dropna().unique().tolist()
        if isinstance(value, pl.Series):
            return value.drop_nulls().unique().to_list()
        return value

    async def resolve_series_async(self, value: Any) -> Any:
        if isinstance(value, dd.Series):
            series = await asyncio.to_thread(value.compute)
            return series.dropna().unique().tolist()
        return self.resolve_series(value)


class DaskFrameStrategy(FrameStrategy):
    engine: ResolvedReturnType = "dask"
    loader_return_type: LoaderReturnType = "dask"

    def normalize(self, frame: FrameResult) -> dd.DataFrame:
        if isinstance(frame, dd.DataFrame):
            return frame
        return dd.from_pandas(_to_pandas(frame), npartitions=1)

    def concat(self, frames: list[FrameResult]) -> dd.DataFrame:
        return dd.concat([self.normalize(frame) for frame in frames], ignore_unknown_divisions=True)

    def has_any_rows(self, frame: FrameResult) -> bool:
        dataframe = self.normalize(frame)
        if len(dataframe.columns) == 0:
            return False
        try:
            head = dataframe.head(1, npartitions=1, compute=True)
            return len(head.index) > 0
        except Exception:
            _log.debug("Failed to check has_any_rows for Dask frame", exc_info=True)
            return False

    def rename_columns(self, frame: FrameResult, rename_map: dict[str, str]) -> dd.DataFrame:
        dataframe = self.normalize(frame)
        return dataframe.rename(columns=rename_map) if rename_map else dataframe

    def select_columns(self, frame: FrameResult, columns: list[str]) -> dd.DataFrame:
        return self.normalize(frame)[columns]

    def apply_column_names(self, frame: FrameResult, column_names: list[str]) -> dd.DataFrame:
        dataframe = self.normalize(frame)
        current = list(dataframe.columns)
        if len(column_names) != len(current):
            raise ValueError(
                f"column_names length ({len(column_names)}) does not match DataFrame column count ({len(current)}): {current}."
            )
        return dataframe.rename(columns=dict(zip(current, column_names)))

    def apply_options(
        self,
        frame: FrameResult,
        params: DataFrameParams,
        opts: DataFrameOptions,
    ) -> dd.DataFrame:
        dataframe = self.normalize(frame)
        if opts.sort_field:
            dataframe = dataframe.sort_values(opts.sort_field)
        if opts.duplicate_expr is not None:
            dataframe = dataframe.drop_duplicates(
                subset=opts.duplicate_expr,
                keep=opts.duplicate_keep,
            )
        if opts.group_by_expr is not None and opts.group_expr is not None:
            dataframe = dataframe.groupby(opts.group_by_expr).agg(opts.group_expr)
        if params.datetime_index:
            dataframe[params.datetime_index] = dd.to_datetime(dataframe[params.datetime_index])
            dataframe = dataframe.set_index(params.datetime_index)
        elif params.index_col:
            dataframe = dataframe.set_index(params.index_col)
        return dataframe

    def persist(self, frame: FrameResult) -> dd.DataFrame:
        return self.normalize(frame).persist()


class PandasFrameStrategy(FrameStrategy):
    engine: ResolvedReturnType = "pandas"
    loader_return_type: LoaderReturnType = "pandas"

    def normalize(self, frame: FrameResult) -> pd.DataFrame:
        return _to_pandas(frame)

    def concat(self, frames: list[FrameResult]) -> pd.DataFrame:
        return pd.concat([self.normalize(frame) for frame in frames], ignore_index=True)

    def has_any_rows(self, frame: FrameResult) -> bool:
        dataframe = self.normalize(frame)
        return len(dataframe.columns) > 0 and len(dataframe.index) > 0

    def rename_columns(self, frame: FrameResult, rename_map: dict[str, str]) -> pd.DataFrame:
        dataframe = self.normalize(frame)
        return dataframe.rename(columns=rename_map) if rename_map else dataframe

    def select_columns(self, frame: FrameResult, columns: list[str]) -> pd.DataFrame:
        return self.normalize(frame)[columns]

    def apply_column_names(self, frame: FrameResult, column_names: list[str]) -> pd.DataFrame:
        dataframe = self.normalize(frame)
        current = list(dataframe.columns)
        if len(column_names) != len(current):
            raise ValueError(
                f"column_names length ({len(column_names)}) does not match DataFrame column count ({len(current)}): {current}."
            )
        return dataframe.rename(columns=dict(zip(current, column_names)))

    def apply_options(
        self,
        frame: FrameResult,
        params: DataFrameParams,
        opts: DataFrameOptions,
    ) -> pd.DataFrame:
        dataframe = self.normalize(frame)
        if opts.sort_field:
            dataframe = dataframe.sort_values(opts.sort_field)
        if opts.duplicate_expr is not None:
            dataframe = dataframe.drop_duplicates(
                subset=opts.duplicate_expr,
                keep=opts.duplicate_keep,
            )
        if opts.group_by_expr is not None and opts.group_expr is not None:
            dataframe = dataframe.groupby(opts.group_by_expr).agg(opts.group_expr)
        if params.datetime_index:
            dataframe[params.datetime_index] = pd.to_datetime(dataframe[params.datetime_index])
            dataframe = dataframe.set_index(params.datetime_index)
        elif params.index_col:
            dataframe = dataframe.set_index(params.index_col)
        return dataframe


class ArrowFrameStrategy(FrameStrategy):
    engine: ResolvedReturnType = "arrow"
    loader_return_type: LoaderReturnType = "arrow"

    def normalize(self, frame: FrameResult) -> pa.Table:
        return _to_arrow(frame)

    def concat(self, frames: list[FrameResult]) -> pa.Table:
        normalized = [self.normalize(frame) for frame in frames]
        return pa.concat_tables(normalized)

    def has_any_rows(self, frame: FrameResult) -> bool:
        table = self.normalize(frame)
        return table.num_rows > 0 and len(table.column_names) > 0

    def rename_columns(self, frame: FrameResult, rename_map: dict[str, str]) -> pa.Table:
        table = self.normalize(frame)
        if not rename_map:
            return table
        current_names = list(table.column_names)
        new_names = [rename_map.get(name, name) for name in current_names]
        return table.rename_columns(new_names)

    def select_columns(self, frame: FrameResult, columns: list[str]) -> pa.Table:
        return self.normalize(frame).select(columns)

    def apply_column_names(self, frame: FrameResult, column_names: list[str]) -> pa.Table:
        table = self.normalize(frame)
        if len(column_names) != len(table.column_names):
            raise ValueError(
                f"column_names length ({len(column_names)}) does not match DataFrame column count ({len(table.column_names)}): {table.column_names}."
            )
        return table.rename_columns(column_names)

    def apply_options(
        self,
        frame: FrameResult,
        params: DataFrameParams,
        opts: DataFrameOptions,
    ) -> pa.Table:
        if params.datetime_index or params.index_col:
            raise ValueError("Arrow return_type does not support index_col or datetime_index.")
        duplicate_expr = opts.duplicate_expr
        if isinstance(duplicate_expr, str):
            duplicate_expr = [duplicate_expr]
        return apply_table_options(
            self.normalize(frame),
            sort_field=opts.sort_field,
            duplicate_expr=duplicate_expr,
            duplicate_keep=_normalize_duplicate_keep(opts.duplicate_keep),
            group_by_expr=opts.group_by_expr,
            group_expr=opts.group_expr,
        )


class PolarsFrameStrategy(FrameStrategy):
    engine: ResolvedReturnType = "polars"
    loader_return_type: LoaderReturnType = "arrow"

    def normalize(self, frame: FrameResult) -> pl.DataFrame:
        return _to_polars(frame)

    def concat(self, frames: list[FrameResult]) -> pl.DataFrame:
        return pl.concat([self.normalize(frame) for frame in frames], how="vertical_relaxed")

    def has_any_rows(self, frame: FrameResult) -> bool:
        dataframe = self.normalize(frame)
        return bool(dataframe.columns) and dataframe.height > 0

    def rename_columns(self, frame: FrameResult, rename_map: dict[str, str]) -> pl.DataFrame:
        dataframe = self.normalize(frame)
        return dataframe.rename(rename_map) if rename_map else dataframe

    def select_columns(self, frame: FrameResult, columns: list[str]) -> pl.DataFrame:
        return self.normalize(frame).select(columns)

    def apply_column_names(self, frame: FrameResult, column_names: list[str]) -> pl.DataFrame:
        dataframe = self.normalize(frame)
        current = list(dataframe.columns)
        if len(column_names) != len(current):
            raise ValueError(
                f"column_names length ({len(column_names)}) does not match DataFrame column count ({len(current)}): {current}."
            )
        return dataframe.rename(dict(zip(current, column_names)))

    def apply_options(
        self,
        frame: FrameResult,
        params: DataFrameParams,
        opts: DataFrameOptions,
    ) -> pl.DataFrame:
        if params.datetime_index or params.index_col:
            raise ValueError("Polars return_type does not support index_col or datetime_index.")
        dataframe = self.normalize(frame)
        if opts.sort_field:
            dataframe = dataframe.sort(opts.sort_field)
        if opts.duplicate_expr is not None:
            subset = [opts.duplicate_expr] if isinstance(opts.duplicate_expr, str) else opts.duplicate_expr
            dataframe = dataframe.unique(subset=subset, keep=_normalize_duplicate_keep(opts.duplicate_keep))
        if opts.group_by_expr is not None and opts.group_expr is not None:
            group_keys = [opts.group_by_expr] if isinstance(opts.group_by_expr, str) else list(opts.group_by_expr)
            dataframe = dataframe.group_by(group_keys).agg(
                _build_polars_group_exprs(dataframe, group_keys, opts.group_expr)
            )
        return dataframe


_FRAME_STRATEGIES: dict[ResolvedReturnType, FrameStrategy] = {
    "dask": DaskFrameStrategy(),
    "pandas": PandasFrameStrategy(),
    "arrow": ArrowFrameStrategy(),
    "polars": PolarsFrameStrategy(),
}


def get_frame_strategy(engine: ResolvedReturnType) -> FrameStrategy:
    try:
        return _FRAME_STRATEGIES[engine]
    except KeyError as exc:
        raise ValueError(f"Unsupported frame engine: {engine!r}") from exc
