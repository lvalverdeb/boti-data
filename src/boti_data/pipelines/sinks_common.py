"""Shared types, frame prep, and staged-write helpers for pipeline sinks.

Split out of sinks.py purely for line-count headroom. Re-exported from
sinks.py so every existing import path keeps working unchanged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Union

import dask.dataframe as dd
import fsspec
import pandas as pd
import polars as pl
import pyarrow as pa

from boti_data.parquet import ParquetDataConfig, ParquetReader

type FrameResult = Union[pd.DataFrame, dd.DataFrame, pa.Table, pl.DataFrame]
type ParquetDestination = Union[ParquetReader, ParquetDataConfig, Mapping[str, Any]]


def _dask_frame_from_pandas(frame: pd.DataFrame) -> dd.DataFrame:
    rows = len(frame) or 1
    npartitions = max(1, rows // 50_000 + 1)
    return dd.from_pandas(frame, npartitions=npartitions)


_TO_DASK_CONVERTERS: list[tuple[type, Callable[[Any], dd.DataFrame]]] = [
    (dd.DataFrame, lambda frame: frame),
    (pd.DataFrame, _dask_frame_from_pandas),
    (pa.Table, lambda frame: dd.from_pandas(frame.to_pandas(), npartitions=1)),
    (pl.DataFrame, lambda frame: dd.from_pandas(frame.to_pandas(), npartitions=1)),
]


def to_dask_frame(frame: FrameResult) -> dd.DataFrame:
    for frame_type, converter in _TO_DASK_CONVERTERS:
        if isinstance(frame, frame_type):
            return converter(frame)
    raise TypeError(f"Unsupported frame type for sink writes: {type(frame)!r}")


def _derive_partition_date_column(
    frame: dd.DataFrame, date_field: str | None, sink_name: str
) -> dd.DataFrame:
    if date_field is None:
        raise ValueError(f"{sink_name} cannot derive partition_date without date_field.")
    if date_field not in frame.columns:
        raise ValueError(
            f"{sink_name} expected date_field={date_field!r} in loaded frame columns "
            f"but only found {list(frame.columns)!r}."
        )
    pre_assign_dtypes = frame.dtypes
    parsed_dates = dd.to_datetime(frame[date_field], errors="coerce")
    frame = frame.assign(partition_date=parsed_dates.dt.date.astype(str))
    # dask-expr's query optimizer can re-derive _meta for the whole frame when
    # .assign() adds a column, silently dropping explicit per-column dtype hints
    # (e.g. bool cast via map_partitions upstream) in favor of generic `object`.
    # Restore any dtype that changed as an unintended side effect of the assign.
    changed_dtypes = {
        column: dtype
        for column, dtype in pre_assign_dtypes.items()
        if column in frame.columns and frame.dtypes[column] != dtype
    }
    if changed_dtypes:
        frame = frame.astype(changed_dtypes)
    return frame


def _validate_meta_matches_real_dtypes(frame: dd.DataFrame, *, sink_name: str) -> None:
    """Guard against dask-expr's meta/real dtype divergence (wishlist #2/#7).

    A bare `dd.to_datetime()`/`pd.to_numeric()` reassignment or an unannotated
    `map_partitions()` call anywhere upstream in a pipeline -- not just inside
    this module's own partition_date derivation -- can silently corrupt an
    *unrelated* column's synthetic `_meta_nonempty` sample without touching
    the real per-partition data. Left unchecked, that divergence doesn't fail
    here: it surfaces later as an opaque pyarrow `ArrowInvalid` at write time
    with nothing pointing back to the real cause. Catch it against one real
    computed partition before any sink commits to a write.
    """
    meta_dtypes = frame.dtypes
    try:
        real_dtypes = frame.head(1, npartitions=-1, compute=True).dtypes
    except ValueError:
        return  # empty frame -- nothing real to validate meta against
    mismatched = {
        column: (meta_dtypes[column], real_dtypes[column])
        for column in real_dtypes.index
        if meta_dtypes[column] != real_dtypes[column]
    }
    if not mismatched:
        return
    details = ", ".join(
        f"{column!r} (meta={meta!s}, real={real!s})" for column, (meta, real) in mismatched.items()
    )
    raise ValueError(
        f"{sink_name} detected a dask meta/real dtype mismatch before write: {details}. "
        "This usually means a column elsewhere in the pipeline was reassigned via a bare "
        "dd.to_datetime()/pd.to_numeric() call or an unannotated map_partitions(), which can "
        "silently corrupt _meta_nonempty for unrelated columns. Pass an explicit meta= to "
        "map_partitions() (or route the coercion through it) instead of reassigning via a "
        "bare top-level pandas/dask function."
    )


def prepare_partitioned_frame(
    frame: dd.DataFrame,
    *,
    partition_on: Sequence[str] | None,
    date_field: str | None,
    sink_name: str,
) -> dd.DataFrame:
    if partition_on:
        partition_columns = list(partition_on)
        missing = [name for name in partition_columns if name not in frame.columns]
        if "partition_date" in missing:
            frame = _derive_partition_date_column(frame, date_field, sink_name)
            missing = [name for name in partition_columns if name not in frame.columns]

        if missing:
            raise ValueError(
                f"{sink_name} partition columns must already exist in the loaded frame or be "
                f"derivable from date_field. Missing columns: {missing!r}."
            )

    _validate_meta_matches_real_dtypes(frame, sink_name=sink_name)
    return frame


def _validate_storage_path(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("storage_path must be specified.")
    return normalized.rstrip("/")


def _rm_recursive(fs: fsspec.AbstractFileSystem, path: str) -> None:
    try:
        fs.rm(path, recursive=True)
    except TypeError:
        fs.rm(path)


def _write_with_staging(
    *,
    fs: fsspec.AbstractFileSystem,
    target_path: str,
    overwrite: bool,
    write_fn: Callable[[str], list[str]],
) -> list[str]:
    """Write sink output, protecting existing data when overwriting.

    When *overwrite* is set and the target already exists, the new output is
    fully computed into a ``.staging`` sibling first, and the previous output
    is removed only after the write succeeded — so a failed compute can no
    longer destroy the existing dataset. The remove-then-rename swap itself
    is not atomic: a crash between the two steps leaves the new data in the
    staging directory, which the next overwrite run cleans up.
    """
    if not overwrite or not fs.exists(target_path):
        return write_fn(target_path)

    staging_path = f"{target_path.rstrip('/')}.staging"
    if fs.exists(staging_path):
        # Leftover from a previous crashed run.
        _rm_recursive(fs, staging_path)

    files = write_fn(staging_path)

    try:
        _rm_recursive(fs, target_path)
        try:
            fs.mv(staging_path, target_path, recursive=True)
        except TypeError:
            fs.mv(staging_path, target_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to swap staged sink output into {target_path!r}. "
            f"The newly written data is intact at {staging_path!r}."
        ) from exc

    return [file.replace(staging_path, target_path.rstrip("/"), 1) for file in files]


@dataclass(slots=True)
class SinkWriteResult:
    """Result returned by pipeline sinks after a write operation."""

    path: str
    files: tuple[str, ...] = ()


class PipelineSink(Protocol):
    """Minimal contract for pipeline write targets."""

    # Not a copy-pasted twin: this is a Protocol interface stub (body is `...`).
    # The real implementations (CsvSink/JsonlSink/ParquetSink) already share
    # _AsyncWriteViaThreadMixin, whose awrite() is a thin asyncio.to_thread()
    # wrapper around write().
    # spaghetti-ignore[sync-async-duplication]: see above
    def write(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult: ...

    async def awrite(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult: ...


class _AsyncWriteViaThreadMixin:
    """Shared ``awrite()`` for sinks whose ``write()`` is itself blocking I/O."""

    async def awrite(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult:
        return await asyncio.to_thread(
            self.write,
            frame,
            date_field=date_field,
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
        )
