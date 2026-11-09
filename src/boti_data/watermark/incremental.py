from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import pyarrow as pa

from boti_data._engines import is_dask_frame, is_pandas_frame, is_polars_frame
from boti_data._engines import polars as _pl

if TYPE_CHECKING:
    import dask.dataframe as dd
    import polars as pl

    FrameLike = pd.DataFrame | dd.DataFrame | pl.DataFrame | pa.Table
else:
    FrameLike = Any


@dataclass
class IncrementalResult:
    """Result returned by incremental load operations.

    Attributes:
        frame: The loaded data (may be empty).
        watermark_field: The field used as the watermark column.
        previous_watermark: The watermark value before this load, or ``None``.
        current_watermark: The watermark value after this load, or ``None``
            when the result frame is empty.
        records_loaded: Number of rows in *frame*.
        watermark_committed: Whether the new watermark was persisted.
    """

    frame: FrameLike
    watermark_field: str
    previous_watermark: Any | None
    current_watermark: Any | None = field(compare=False)
    records_loaded: int = field(compare=False)
    watermark_committed: bool = field(compare=False)

    def __bool__(self) -> bool:
        return self.records_loaded > 0


def build_incremental_filters(
    *,
    watermark_field: str,
    watermark_value: Any,
    operator: Literal["gt", "gte"] = "gt",
) -> dict[str, Any]:
    """Build a filter dict that selects rows strictly after the watermark.

    The returned dict is suitable for passing as ``filters`` to
    :meth:`DataHelper.load` or the gateway's ``load`` method.

    Args:
        watermark_field: Column name to filter on.
        watermark_value: The reference value (rows strictly after this).
        operator: ``"gt"`` (strictly greater than) or ``"gte"`` (greater
            than or equal).

    Returns:
        A single-key filter dict, e.g. ``{"updated_at__gt": ...}``.
    """
    suffix = "__gt" if operator == "gt" else "__gte"
    return {f"{watermark_field}{suffix}": watermark_value}


def _advance_dask(frame: dd.DataFrame, watermark_field: str) -> Any | None:
    if frame.npartitions == 0 or len(frame) == 0:
        return None
    result = frame[watermark_field].max()
    if hasattr(result, "compute"):
        result = result.compute()
    if _is_null(result):
        return None
    if hasattr(result, "item"):
        result = result.item()
    return result


def _advance_pandas(frame: pd.DataFrame, watermark_field: str) -> Any | None:
    if len(frame.index) == 0:
        return None
    result = frame[watermark_field].max()
    if _is_null(result):
        return None
    if hasattr(result, "item"):
        result = result.item()
    return result


def _advance_polars(frame: pl.DataFrame, watermark_field: str) -> Any | None:
    if frame.is_empty():
        return None
    result = frame.select(_pl().max(watermark_field)).item()
    if _is_null(result):
        return None
    return result


def _advance_pyarrow(frame: pa.Table, watermark_field: str) -> Any | None:
    if frame.num_rows == 0:
        return None
    column_index = frame.schema.get_field_index(watermark_field)
    non_null = (
        [v for v in frame.column(column_index).to_pylist() if v is not None]
        if column_index >= 0
        else []
    )
    if not non_null:
        return None
    return max(non_null)


def _is_arrow_table(obj: Any) -> bool:
    return isinstance(obj, pa.Table)


_ADVANCE_WATERMARK_DISPATCH: list[
    tuple[Callable[[Any], bool], Callable[[Any, str], Any | None]]
] = [
    (is_dask_frame, _advance_dask),
    (is_pandas_frame, _advance_pandas),
    (is_polars_frame, _advance_polars),
    (_is_arrow_table, _advance_pyarrow),
]


def advance_watermark(
    frame: FrameLike,
    *,
    watermark_field: str,
) -> Any | None:
    """Determine the next watermark value from the given frame.

    Returns the maximum value of *watermark_field* in *frame*, or
    ``None`` if the frame is empty or the column is all-null.

    Supports pandas, Dask, Polars, and PyArrow frames.
    """
    for frame_type, advance in _ADVANCE_WATERMARK_DISPATCH:
        if frame_type(frame):
            return advance(frame, watermark_field)
    return None


def _is_null(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or (hasattr(value, "isna") and value is pd.NA)
    )


__all__ = [
    "IncrementalResult",
    "advance_watermark",
    "build_incremental_filters",
]
