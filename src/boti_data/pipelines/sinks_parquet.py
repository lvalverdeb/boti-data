"""Parquet pipeline sink.

Split out of sinks.py purely for line-count headroom. Re-exported from
sinks.py so every existing import path keeps working unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from boti.core.lifecycle import LifecycleCore
from boti.core.lifecycle_pickle import PicklableLifecycleCoreMixin
from boti_dask import safe_persist

from boti_data.parquet import ParquetDataConfig, ParquetReader
from boti_data.pipelines.sinks_common import (
    FrameResult,
    ParquetDestination,
    SinkWriteResult,
    _AsyncWriteViaThreadMixin,
    _write_with_staging,
    prepare_partitioned_frame,
    to_dask_frame,
)


class ParquetSink(_AsyncWriteViaThreadMixin, PicklableLifecycleCoreMixin, LifecycleCore):
    """Write a frame to a parquet dataset directory and expose the reader used for reloads."""

    def __init__(
        self,
        destination: ParquetDestination,
        *,
        partition_on: Sequence[str] | None = ("partition_date",),
    ) -> None:
        self.partition_on = list(partition_on) if partition_on is not None else None
        self.reader = self._coerce_destination(destination)
        if self.reader.parquet_filename is not None:
            raise ValueError(
                "ParquetSink destination must point to a parquet dataset directory; "
                "parquet_filename is not supported for materialized write targets."
            )
        super().__init__()

    def __enter__(self) -> ParquetSink:
        super().__enter__()
        self.reader.__enter__()
        return self

    async def __aenter__(self) -> ParquetSink:
        await super().__aenter__()
        await self.reader.__aenter__()
        return self

    def _cleanup(self) -> None:
        self.reader.close()

    async def _acleanup(self) -> None:
        await self.reader.aclose()

    @staticmethod
    def _coerce_destination(destination: ParquetDestination) -> ParquetReader:
        if isinstance(destination, ParquetReader):
            return destination
        if isinstance(destination, ParquetDataConfig):
            return ParquetReader(destination)
        if isinstance(destination, Mapping):
            payload = dict(destination)
            payload.setdefault("backend", "parquet")
            return ParquetReader(payload)
        raise TypeError(
            "ParquetSink destination must be a ParquetReader, ParquetDataConfig, or mapping. "
            f"Got {type(destination)!r}."
        )

    def write(
        self,
        frame: FrameResult,
        *,
        date_field: str | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
    ) -> SinkWriteResult:
        """Write *frame* to the configured parquet destination.

        ``overwrite=True`` replaces the *entire* target directory (all
        partitions/dates), not just the partition(s) present in *frame* — see
        :func:`_write_with_staging`. For incremental per-date writes, do not
        rely on ``overwrite=True`` to mean "overwrite this date only".
        """
        ddf = to_dask_frame(frame)
        ddf = prepare_partitioned_frame(
            ddf,
            partition_on=self.partition_on,
            date_field=date_field,
            sink_name="ParquetSink",
            validate_arrow_convertible=True,
        )
        if persist:
            ddf = safe_persist(ddf)

        fs = self.reader.resource.require_fs()
        target_path = cast(str, self.reader.parquet_storage_path)

        def _write(directory: str) -> list[str]:
            params: dict[str, Any] = {
                "path": directory,
                "engine": "pyarrow",
                "write_index": write_index,
                "filesystem": fs,
            }
            if self.partition_on:
                params["partition_on"] = list(self.partition_on)
            ddf.to_parquet(**params)
            return sorted(
                self._restore_protocol(path) for path in fs.glob(f"{directory.rstrip('/')}/*")
            )

        files = _write_with_staging(
            fs=fs, target_path=target_path, overwrite=overwrite, write_fn=_write
        )
        return SinkWriteResult(path=target_path, files=tuple(files))

    @staticmethod
    def _restore_protocol(path: str) -> str:
        return str(path)


# Not a copy-pasted twin: `with`/`async with` + sink.write()/await sink.awrite()
# is the irreducible sync/async difference; there's no further shared logic to
# hoist for what's just a construct+write+close convenience.
# spaghetti-ignore[sync-async-duplication]: see above
def write_parquet(
    destination: ParquetDestination,
    frame: FrameResult,
    *,
    partition_on: Sequence[str] | None = ("partition_date",),
    **kwargs: Any,
) -> SinkWriteResult:
    """One-shot convenience: construct a ParquetSink, write *frame*, and close it.

    A ``ParquetSink`` is typically built, used for exactly one ``write()``,
    and discarded — forgetting ``with ParquetSink(...) as sink:`` leaks the
    underlying ``DataGateway``/``ParquetDataResource``, surfaced only as a
    background GC warning well after the fact, not at the call site. Callers
    who genuinely reuse one ``ParquetSink`` instance across multiple writes
    should keep using the class directly instead.

    ``**kwargs`` are forwarded to :meth:`ParquetSink.write` (``date_field``,
    ``write_index``, ``overwrite``, ``persist``).
    """
    with ParquetSink(destination, partition_on=partition_on) as sink:
        return sink.write(frame, **kwargs)


async def awrite_parquet(
    destination: ParquetDestination,
    frame: FrameResult,
    *,
    partition_on: Sequence[str] | None = ("partition_date",),
    **kwargs: Any,
) -> SinkWriteResult:
    """Async variant of :func:`write_parquet`."""
    async with ParquetSink(destination, partition_on=partition_on) as sink:
        return await sink.awrite(frame, **kwargs)
