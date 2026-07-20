"""PyArrow compute kernels: type coercion and comparison operators.

Split out of arrow_kernels.py purely for god-module headroom. Re-exported
from arrow_kernels.py so every existing import path keeps working unchanged.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc


def ensure_string_array(arr: pa.ChunkedArray | pa.Array) -> pa.ChunkedArray:
    """Ensure an array is string-typed, casting if necessary."""
    if isinstance(arr, pa.Array):
        arr = pa.chunked_array([arr])
    if not pa.types.is_string(arr.type) and not pa.types.is_large_string(arr.type):
        return pc.cast(arr, pa.string())
    return arr


def ensure_chunked(arr: pa.Array | pa.ChunkedArray) -> pa.ChunkedArray:
    """Wrap a scalar or Array into a ChunkedArray."""
    if isinstance(arr, pa.Array):
        return pa.chunked_array([arr])
    return arr


def exact_kernel(column: pa.ChunkedArray, value: Any) -> pa.ChunkedArray:
    """column == value"""
    return pc.equal(column, pa.scalar(value))


def not_exact_kernel(column: pa.ChunkedArray, value: Any) -> pa.ChunkedArray:
    """column != value"""
    return pc.not_equal(column, pa.scalar(value))


def gt_kernel(column: pa.ChunkedArray, value: Any) -> pa.ChunkedArray:
    """column > value"""
    return pc.greater(column, pa.scalar(value))


def gte_kernel(column: pa.ChunkedArray, value: Any) -> pa.ChunkedArray:
    """column >= value"""
    return pc.greater_equal(column, pa.scalar(value))


def lt_kernel(column: pa.ChunkedArray, value: Any) -> pa.ChunkedArray:
    """column < value"""
    return pc.less(column, pa.scalar(value))


def lte_kernel(column: pa.ChunkedArray, value: Any) -> pa.ChunkedArray:
    """column <= value"""
    return pc.less_equal(column, pa.scalar(value))


def in_kernel(column: pa.ChunkedArray, values: list[Any]) -> pa.ChunkedArray:
    """column IN (values...)"""
    return pc.is_in(column, value_set=pa.array(values))


def not_in_kernel(column: pa.ChunkedArray, values: list[Any]) -> pa.ChunkedArray:
    """column NOT IN (values...)"""
    return pc.invert(pc.is_in(column, value_set=pa.array(values)))


def range_kernel(column: pa.ChunkedArray, lower: Any, upper: Any) -> pa.ChunkedArray:
    """lower <= column <= upper"""
    return pc.and_(
        pc.greater_equal(column, pa.scalar(lower)),
        pc.less_equal(column, pa.scalar(upper)),
    )


def isnull_kernel(column: pa.ChunkedArray, is_null: bool = True) -> pa.ChunkedArray:
    """column IS NULL (or IS NOT NULL)"""
    mask = pc.is_null(column)
    return mask if is_null else pc.invert(mask)
