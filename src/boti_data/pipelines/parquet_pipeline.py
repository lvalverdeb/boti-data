"""Parquet-specific pipeline: materialize + incremental watermark support.

Split out of base.py purely for line-count headroom: ParquetPipeline is a
self-contained subclass of SinkPipeline, and nothing outside pipelines/
references its supporting types directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from boti_data.pipelines.base import PipelineDestination, PipelineSource, SinkPipeline
from boti_data.pipelines.sinks import ParquetSink
from boti_data.watermark import FileWatermarkStore, WatermarkStore, advance_watermark


@dataclass(slots=True)
class ParquetMaterializationResult:
    """Result returned by :meth:`ParquetPipeline.materialize` / ``amaterialize``.

    Attributes:
        path: Dataset directory written by parquet materialization.
        frame: Optional parquet reload result when ``reload=True`` was requested.
    """

    path: str
    frame: Any | None = None

    @property
    def reloaded(self) -> bool:
        return self.frame is not None


@dataclass(frozen=True)
class MaterializeWriteOptions:
    """The write/reload knobs shared by materialize()/amaterialize()'s
    sync/async and full/incremental internal twins."""

    reload: bool = False
    reload_options: Mapping[str, Any] | None = None
    write_index: bool = False
    overwrite: bool = True
    persist: bool = False


class ParquetPipeline(SinkPipeline):
    """Materialize `DataHelper` or `HybridDataset` loads into a parquet dataset.

    Adds parquet reload capabilities on top of the generic :class:`SinkPipeline` write flow.
    """

    def __init__(
        self,
        source: PipelineSource,
        destination: PipelineDestination,
        *,
        date_field: str | None = None,
        partition_on: tuple[str, ...] | list[str] | None = ("partition_date",),
    ) -> None:
        self.parquet_sink = ParquetSink(destination, partition_on=partition_on)
        super().__init__(source, self.parquet_sink, date_field=date_field)
        self.reader = self.parquet_sink.reader

    def from_parquet(self, **options: Any) -> Any:
        return self.reader.load(**options)

    async def afrom_parquet(self, **options: Any) -> Any:
        return await self.reader.aload(**options)

    # Not a copy-pasted twin: pure .path-unwrapping delegate to the
    # already-split write()/awrite() pair on SinkPipeline.
    # spaghetti-ignore[sync-async-duplication]: see above
    def to_parquet(
        self,
        *,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
        **load_options: Any,
    ) -> str:
        return self.write(
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
            **load_options,
        ).path

    async def ato_parquet(
        self,
        *,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
        **load_options: Any,
    ) -> str:
        result = await self.awrite(
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
            **load_options,
        )
        return result.path

    @staticmethod
    def _reload_options(reload_options: Mapping[str, Any] | None) -> dict[str, Any]:
        return dict(reload_options or {})

    @classmethod
    def incremental(
        cls,
        source: PipelineSource,
        destination: PipelineDestination,
        *,
        watermark_field: str,
        watermark_store: WatermarkStore | None = None,
        watermark_source: str | None = None,
        initial_value: Any = None,
        date_field: str | None = None,
        partition_on: tuple[str, ...] | list[str] | None = ("partition_date",),
    ) -> ParquetPipeline:
        """Create a :class:`ParquetPipeline` whose source is wired for incremental
        materialization.

        The returned pipeline wraps *source* so that every ``materialize()``
        call pulls only rows where *watermark_field* is greater than the
        last successfully loaded value.  The watermark is persisted in
        *watermark_store* under *watermark_source* (defaults to the source's
        table name or ``"default"``).

        On the first run, if no watermark exists and *initial_value* is set,
        rows are filtered from that value onward.  Otherwise a full load is
        performed.
        """
        pipeline = cls(
            source,
            destination,
            date_field=date_field,
            partition_on=partition_on,
        )
        pipeline._watermark_field = watermark_field
        pipeline._watermark_source = (
            watermark_source
            or (getattr(source, "gateway", None) and getattr(source.gateway, "_table", None))
            or "default"
        )
        pipeline._watermark_store = watermark_store or FileWatermarkStore()
        pipeline._initial_value = initial_value
        return pipeline

    # Not a copy-pasted twin: pure watermark-presence dispatcher delegating to
    # already-split _materialize_full()/_amaterialize_full() and
    # _materialize_incremental()/_amaterialize_incremental() below.
    # spaghetti-ignore[sync-async-duplication]: see above
    def materialize(
        self,
        *,
        reload: bool = False,
        reload_options: Mapping[str, Any] | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
        **load_options: Any,
    ) -> ParquetMaterializationResult:
        options = MaterializeWriteOptions(
            reload=reload,
            reload_options=reload_options,
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
        )
        watermark_field: str | None = getattr(self, "_watermark_field", None)
        if watermark_field is not None:
            return self._materialize_incremental(options, **load_options)
        return self._materialize_full(options, **load_options)

    async def amaterialize(
        self,
        *,
        reload: bool = False,
        reload_options: Mapping[str, Any] | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
        **load_options: Any,
    ) -> ParquetMaterializationResult:
        options = MaterializeWriteOptions(
            reload=reload,
            reload_options=reload_options,
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
        )
        watermark_field: str | None = getattr(self, "_watermark_field", None)
        if watermark_field is not None:
            return await self._amaterialize_incremental(options, **load_options)
        return await self._amaterialize_full(options, **load_options)

    def _materialize_full(
        self, options: MaterializeWriteOptions, **load_options: Any
    ) -> ParquetMaterializationResult:
        result = self.write(
            write_index=options.write_index,
            overwrite=options.overwrite,
            persist=options.persist,
            **load_options,
        )
        frame = None
        if options.reload:
            frame = self.from_parquet(**self._reload_options(options.reload_options))
        return ParquetMaterializationResult(path=result.path, frame=frame)

    async def _amaterialize_full(
        self, options: MaterializeWriteOptions, **load_options: Any
    ) -> ParquetMaterializationResult:
        result = await self.awrite(
            write_index=options.write_index,
            overwrite=options.overwrite,
            persist=options.persist,
            **load_options,
        )
        frame = None
        if options.reload:
            frame = await self.afrom_parquet(**self._reload_options(options.reload_options))
        return ParquetMaterializationResult(path=result.path, frame=frame)

    def _update_watermark_from_frame(self, frame: Any, watermark_field: str) -> None:
        store: WatermarkStore = self._watermark_store
        source: str = self._watermark_source
        current = advance_watermark_incremental(frame, watermark_field)
        if current is not None:
            store.write(source=source, value=current)

    def _prepare_incremental_filters(self, load_options: dict[str, Any]) -> str:
        """Merges watermark-derived filters into ``load_options`` in place.

        Returns the watermark field name for use by the caller.
        """
        watermark_field: str = self._watermark_field
        store: WatermarkStore = self._watermark_store
        source: str = self._watermark_source
        previous = store.read(source=source)
        filters = {}
        if previous is not None:
            filters = {f"{watermark_field}__gt": previous}
        elif self._initial_value is not None:
            filters = {f"{watermark_field}__gt": self._initial_value}
        merged = {**load_options.pop("filters", {}), **filters}
        if merged:
            load_options["filters"] = merged
        return watermark_field

    def _finalize_reload_frame(self, frame: Any, watermark_field: str) -> Any:
        """Discards an empty reloaded frame, else advances the watermark from it."""
        if frame is None:
            return None
        if hasattr(frame, "columns") and len(frame.columns) == 0:
            return None
        self._update_watermark_from_frame(frame, watermark_field)
        return frame

    def _materialize_incremental(
        self, options: MaterializeWriteOptions, **load_options: Any
    ) -> ParquetMaterializationResult:
        watermark_field = self._prepare_incremental_filters(load_options)
        result = self.write(
            write_index=options.write_index,
            overwrite=options.overwrite,
            persist=options.persist,
            **load_options,
        )
        frame = None
        if options.reload:
            if result.files:
                frame = self._finalize_reload_frame(
                    self.from_parquet(**self._reload_options(options.reload_options)),
                    watermark_field,
                )
        else:
            self._update_watermark_from_frame(self.source.load(**load_options), watermark_field)
        return ParquetMaterializationResult(path=result.path, frame=frame)

    async def _amaterialize_incremental(
        self, options: MaterializeWriteOptions, **load_options: Any
    ) -> ParquetMaterializationResult:
        watermark_field = self._prepare_incremental_filters(load_options)
        result = await self.awrite(
            write_index=options.write_index,
            overwrite=options.overwrite,
            persist=options.persist,
            **load_options,
        )
        frame = None
        if options.reload:
            if result.files:
                frame = self._finalize_reload_frame(
                    await self.afrom_parquet(**self._reload_options(options.reload_options)),
                    watermark_field,
                )
        else:
            self._update_watermark_from_frame(
                await self.source.aload(**load_options), watermark_field
            )
        return ParquetMaterializationResult(path=result.path, frame=frame)


def advance_watermark_incremental(frame: Any, watermark_field: str) -> Any | None:
    return advance_watermark(frame, watermark_field=watermark_field)


__all__ = ["MaterializeWriteOptions", "ParquetMaterializationResult", "ParquetPipeline"]
