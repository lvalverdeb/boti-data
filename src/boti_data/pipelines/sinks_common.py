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


def _arrow_convertible(sample: pd.DataFrame) -> bool:
    try:
        pa.Table.from_pandas(sample, preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return False
    return True


def _validate_meta_object_columns_are_arrow_convertible(
    frame: dd.DataFrame, *, sink_name: str
) -> None:
    """Guard against dask's meta_nonempty() sentinel-object corruption (wishlist #10).

    Distinct from `_validate_meta_matches_real_dtypes()` above: that check
    catches meta/real *dtype* divergence, but a genuine ``object``-dtype
    column (e.g. a nested/struct parquet column, or one column of two
    otherwise-identical branches merged via `HybridDataset` where the other
    is entirely null in this slice) can have meta and real dtype agree
    (both ``object``) while still breaking PyArrow at write time. On
    pandas>=3.0, dask's `_nonempty_series()` has no per-partition data to
    sample from `frame._meta` (always 0 rows) and falls back to a bare
    `object()` placeholder for any generic object dtype -- a value with no
    representable Arrow type. Left unchecked, that surfaces as an opaque
    `pyarrow.lib.ArrowInvalid: Could not convert <object object at 0x...>`
    at write time with nothing pointing back to the real column. Catch it
    against the synthesized sample before any sink commits to a write.
    """
    object_columns = [
        column for column, dtype in frame.dtypes.items() if pd.api.types.is_object_dtype(dtype)
    ]
    if not object_columns:
        return
    sample = frame._meta_nonempty
    bad_columns = [column for column in object_columns if not _arrow_convertible(sample[[column]])]
    if not bad_columns:
        return
    raise ValueError(
        f"{sink_name} detected object-dtype column(s) with no PyArrow-representable "
        f"value in dask's synthesized meta sample: {bad_columns!r}. This typically "
        "happens with a nested/struct-typed column, or one entirely null in one branch "
        "of a concat/merge (e.g. HybridDataset's historical vs live branches) while a "
        "sibling branch has real values -- dask can't derive a placeholder PyArrow can "
        "place. Project the load down to only the columns actually needed, or cast the "
        "column to an explicit scalar dtype before writing."
    )


def prepare_partitioned_frame(
    frame: dd.DataFrame,
    *,
    partition_on: Sequence[str] | None,
    date_field: str | None,
    sink_name: str,
    validate_arrow_convertible: bool = False,
) -> dd.DataFrame:
    """Derive partition columns and validate the frame's dask meta before a write.

    ``validate_arrow_convertible`` gates
    `_validate_meta_object_columns_are_arrow_convertible()`: that guard
    exists for PyArrow's schema-inference step, which only ``ParquetSink``
    goes through -- ``CsvSink``/``JsonlSink`` write each partition's real
    data straight through pandas' own ``to_csv``/``to_json`` and would
    reject a perfectly writable object column for a failure mode they never
    hit. `_validate_meta_matches_real_dtypes()` above stays unconditional:
    a meta/real dtype mismatch is a signal of upstream corruption regardless
    of sink type, not a PyArrow-specific concern.
    """
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
    if validate_arrow_convertible:
        _validate_meta_object_columns_are_arrow_convertible(frame, sink_name=sink_name)
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


def _rebase_staged_path(
    fs: fsspec.AbstractFileSystem,
    path: str,
    staging_path: str,
    target_root: str,
) -> str:
    """Rewrite *path* from the staging tree to the target tree.

    ``glob()``/``find()`` on most fsspec backends (S3, memory, ...) return
    protocol-stripped paths (``bucket/key``) even when queried with a
    scheme-prefixed directory (``s3://bucket/key``) -- but *staging_path*/
    *target_root* here are whatever the caller originally passed in, which
    for a ``Datasources``-resolved destination is scheme-prefixed. A naive
    ``path.replace(staging_path, target_root)`` then silently no-ops (no
    substring match, string returned unchanged) instead of raising, so both
    sides are normalized through ``_strip_protocol()`` first to guarantee the
    replace actually matches regardless of which form *path* arrived in.
    """
    normalized_path = fs._strip_protocol(path)
    normalized_staging = fs._strip_protocol(staging_path)
    normalized_target = fs._strip_protocol(target_root)
    return normalized_path.replace(normalized_staging, normalized_target, 1)


def _move_staged_files(
    fs: fsspec.AbstractFileSystem,
    staged_files: Sequence[str],
    staging_path: str,
    target_root: str,
) -> None:
    """Move each enumerated staged file individually into *target_root*.

    Deliberately avoids a single recursive ``fs.mv(staging_path, target_root,
    recursive=True)``. On prefix-only S3-compatible backends (e.g. MinIO),
    ``expand_path(recursive=True)`` can return the bare directory prefix
    itself as a phantom "file" alongside the real objects -- no object
    actually exists at that literal key -- and ``mv()`` forces
    ``on_error="raise"`` on the resulting copy, so that one phantom entry
    404s the entire swap even though every real file would have copied fine.
    Moving the already-known file list one by one never expands a directory,
    so the phantom entry never gets a chance to appear.
    """
    for staged_file in staged_files:
        destination = _rebase_staged_path(fs, staged_file, staging_path, target_root)
        parent = destination.rsplit("/", 1)[0]
        if parent and parent != destination:
            fs.makedirs(parent, exist_ok=True)
        fs.mv(staged_file, destination)
    if fs.exists(staging_path):
        # Empty leftover directory structure (local-disk backends only --
        # prefix-only stores have nothing left here once every file moved).
        _rm_recursive(fs, staging_path)


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
    target_root = target_path.rstrip("/")

    try:
        _rm_recursive(fs, target_path)
        _move_staged_files(fs, files, staging_path, target_root)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to swap staged sink output into {target_path!r}. "
            f"The newly written data is intact at {staging_path!r}."
        ) from exc

    return [_rebase_staged_path(fs, file, staging_path, target_root) for file in files]


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
