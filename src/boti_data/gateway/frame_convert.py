"""Pure frame-type conversion/shaping helpers shared by the FrameStrategy family.

Split out of frame_strategies.py purely for line-count headroom: these
touch no strategy state, so they move here wholesale as free functions.
``_to_arrow`` is re-imported into frame_strategies.py since other modules
(e.g. dataset/hybrid_support.py) import it from there directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa

from boti_data._engines import is_dask_frame, is_pandas_frame, is_polars_frame
from boti_data._engines import polars as _pl
from boti_data.db.arrow_schema_mapper import arrow_table_to_pandas

if TYPE_CHECKING:
    import dask.dataframe as dd
    import polars as pl

    FrameResult = pd.DataFrame | dd.DataFrame | pa.Table | pl.DataFrame
else:
    FrameResult = Any


def _is_arrow_table(obj: Any) -> bool:
    return isinstance(obj, pa.Table)


_TO_PANDAS_CONVERTERS: list[tuple[Callable[[Any], bool], Callable[[Any], pd.DataFrame]]] = [
    (is_pandas_frame, lambda frame: frame),
    (is_dask_frame, lambda frame: frame.compute()),
    (_is_arrow_table, arrow_table_to_pandas),
    (is_polars_frame, lambda frame: frame.to_pandas()),
]


def _to_pandas(frame: FrameResult) -> pd.DataFrame:
    for frame_type, converter in _TO_PANDAS_CONVERTERS:
        if frame_type(frame):
            return converter(frame)
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _to_arrow(frame: FrameResult) -> pa.Table:
    if isinstance(frame, pa.Table):
        return frame
    if is_dask_frame(frame):
        frame = frame.compute()
    if isinstance(frame, pd.DataFrame):
        return pa.Table.from_pandas(frame, preserve_index=False)
    if is_polars_frame(frame):
        return frame.to_arrow()
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _to_polars(frame: FrameResult) -> pl.DataFrame:
    if is_polars_frame(frame):
        return frame
    return _pl().from_arrow(_to_arrow(frame))


def _normalize_duplicate_keep(value: str | bool) -> str:
    return "none" if value is False else value


def _rename_columns_by_position(dataframe: Any, column_names: list[str]) -> Any:
    """Rename *dataframe*'s columns positionally to *column_names*.

    Shared by strategies whose underlying object supports pandas-style
    ``.rename(columns=mapping)`` (dask, pandas).
    """
    current = list(dataframe.columns)
    if len(column_names) != len(current):
        raise ValueError(
            f"column_names length ({len(column_names)}) does not match DataFrame column count ({len(current)}): {current}."
        )
    return dataframe.rename(columns=dict(zip(current, column_names)))


_POLARS_AGG_METHODS = {
    "first": "first",
    "last": "last",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "mean": "mean",
    "count": "count",
    "len": "count",
    "any": "any",
}


def _resolve_group_aggregations(
    dataframe: pl.DataFrame,
    group_keys: list[str],
    group_expr: str | dict[str, str],
) -> dict[str, str]:
    if isinstance(group_expr, str):
        return {name: group_expr for name in dataframe.columns if name not in set(group_keys)}
    return dict(group_expr)


def _build_polars_group_exprs(
    dataframe: pl.DataFrame,
    group_keys: list[str],
    group_expr: str | dict[str, str],
) -> list[pl.Expr]:
    aggregations = _resolve_group_aggregations(dataframe, group_keys, group_expr)
    exprs: list[pl.Expr] = []
    for column, func in aggregations.items():
        method_name = _POLARS_AGG_METHODS.get(func)
        if method_name is None:
            raise ValueError(f"Unsupported polars aggregation: {func!r}")
        exprs.append(getattr(_pl().col(column), method_name)().alias(column))
    return exprs
