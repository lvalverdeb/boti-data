from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Union

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa

FrameResult = Union[pd.DataFrame, dd.DataFrame, pa.Table, pl.DataFrame]


@dataclass(slots=True)
class AttachmentSpec:
    """Declarative enrichment attachment specification.

    Attributes:
        key: Stable spec identifier used by caller selection.
        required_cols: Base-frame columns required for this attachment to run.
        attachment_fn: Async loader called with resolved kwargs.
        col_to_kwarg: Mapping base column -> attachment kwarg receiving unique values.
        left_on: Base frame merge keys.
        right_on: Attachment frame merge keys.
        drop_cols: Columns dropped after merge (usually right-side join keys).
        max_unique_values: Optional per-spec upper bound for extracted unique values.
    """

    key: str
    required_cols: set[str]
    attachment_fn: Callable[..., Awaitable[FrameResult | None]]
    col_to_kwarg: dict[str, str]
    left_on: list[str]
    right_on: list[str]
    drop_cols: list[str]
    max_unique_values: int | None = None

    def is_applicable(self, available_cols: set[str], *, requested_keys: set[str] | None = None) -> bool:
        if requested_keys is not None and self.key not in requested_keys:
            return False
        return self.required_cols.issubset(available_cols)


__all__ = ["AttachmentSpec"]
