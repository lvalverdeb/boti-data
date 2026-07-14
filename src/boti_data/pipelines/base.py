from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, Union, cast

from boti.core.lifecycle import LifecycleCore
from boti.core.lifecycle_pickle import PicklableLifecycleCoreMixin

from boti_data.dataset import HybridDataset
from boti_data.helper import DataHelper
from boti_data.pipelines.registry import create_sink
from boti_data.pipelines.sinks import (
    FrameResult,
    ParquetDestination,
    ParquetSink,
    PipelineSink,
    SinkWriteResult,
)
from boti_data.watermark import (
    FileWatermarkStore,
    WatermarkStore,
    advance_watermark,
)

PipelineSource: TypeAlias = Union[DataHelper, HybridDataset]
PipelineDestination: TypeAlias = ParquetDestination


class FrameEnricher(Protocol):
    def enrich(
        self, base_frame: FrameResult, *, cols: Sequence[str] | None = None
    ) -> FrameResult: ...

    async def aenrich(
        self,
        base_frame: FrameResult,
        *,
        cols: Sequence[str] | None = None,
    ) -> FrameResult: ...


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


class SinkPipeline(PicklableLifecycleCoreMixin, LifecycleCore):
    """Generic orchestration layer that loads from a source and writes into a sink.

    This class intentionally sits *above* `DataHelper`/`DataGateway`: it orchestrates
    loading and sink writes without expanding the gateway core with write-path concerns.
    """

    def __init__(
        self,
        source: PipelineSource,
        sink: PipelineSink | str,
        *,
        sink_config: Mapping[str, Any] | Any | None = None,
        enricher: FrameEnricher | None = None,
        date_field: str | None = None,
    ) -> None:
        if not isinstance(source, (DataHelper, HybridDataset)):
            raise TypeError(
                "SinkPipeline source must be a DataHelper or HybridDataset instance. "
                f"Got {type(source)!r}."
            )
        self.source = source
        self.date_field = date_field
        self.enricher = enricher
        if isinstance(sink, str):
            if sink_config is None:
                raise ValueError("sink_config is required when sink is provided by name.")
            self.sink = create_sink(sink, sink_config)
        else:
            self.sink = sink
        super().__init__()

    def __enter__(self) -> SinkPipeline:
        super().__enter__()
        self.source.__enter__()
        sink_enter = getattr(self.sink, "__enter__", None)
        if callable(sink_enter):
            sink_enter()
        return self

    async def __aenter__(self) -> SinkPipeline:
        await super().__aenter__()
        await self.source.__aenter__()
        sink_aenter = getattr(self.sink, "__aenter__", None)
        if callable(sink_aenter):
            maybe = sink_aenter()
            if inspect.isawaitable(maybe):
                await cast(Any, maybe)
        return self

    def _cleanup(self) -> None:
        # try/finally: a sink close() failure must not skip closing the
        # source (and vice versa isn't possible — source is always closed
        # last, in the finally block, regardless of how the sink behaves).
        try:
            sink_close = getattr(self.sink, "close", None)
            if callable(sink_close):
                sink_close()
        finally:
            self.source.close()

    async def _acleanup(self) -> None:
        try:
            sink_aclose = getattr(self.sink, "aclose", None)
            if callable(sink_aclose):
                maybe = sink_aclose()
                if inspect.isawaitable(maybe):
                    await cast(Any, maybe)
        finally:
            await self.source.aclose()

    def load(self, **options: Any) -> FrameResult:
        return self.source.load(**options)

    async def aload(self, **options: Any) -> FrameResult:
        return await self.source.aload(**options)

    @staticmethod
    def _prepare_write_options(
        load_options: Mapping[str, Any],
    ) -> tuple[dict[str, Any], Sequence[str] | None]:
        resolved_options = dict(load_options)
        enrich_cols = resolved_options.pop("enrich_cols", None)
        return resolved_options, enrich_cols

    def write(
        self,
        *,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
        **load_options: Any,
    ) -> SinkWriteResult:
        resolved_options, enrich_cols = self._prepare_write_options(load_options)
        frame = self.load(**self._materialization_load_options(resolved_options))
        frame = self._maybe_enrich_sync(frame, cols=enrich_cols)
        return self.sink.write(
            frame,
            date_field=self.date_field,
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
        )

    async def awrite(
        self,
        *,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
        **load_options: Any,
    ) -> SinkWriteResult:
        resolved_options, enrich_cols = self._prepare_write_options(load_options)
        frame = await self.aload(**self._materialization_load_options(resolved_options))
        frame = await self._maybe_enrich_async(frame, cols=enrich_cols)
        return await self.sink.awrite(
            frame,
            date_field=self.date_field,
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
        )

    @staticmethod
    def _materialization_load_options(load_options: Mapping[str, Any]) -> dict[str, Any]:
        resolved = dict(load_options)
        requested_return_type = resolved.get("return_type")
        if requested_return_type not in {None, "dask"}:
            raise ValueError(
                "SinkPipeline materialization requires return_type='dask' so writes remain lazy."
            )
        requested_execution_mode = resolved.get("execution_mode")
        if requested_execution_mode not in {None, "lazy"}:
            raise ValueError("SinkPipeline materialization requires execution_mode='lazy'.")
        if resolved.get("as_pandas"):
            raise ValueError("SinkPipeline materialization does not support as_pandas=True.")
        resolved["return_type"] = "dask"
        resolved["execution_mode"] = "lazy"
        return resolved

    def _maybe_enrich_sync(
        self,
        frame: FrameResult,
        *,
        cols: Sequence[str] | None,
    ) -> FrameResult:
        if self.enricher is None:
            return frame
        return self.enricher.enrich(frame, cols=cols)

    async def _maybe_enrich_async(
        self,
        frame: FrameResult,
        *,
        cols: Sequence[str] | None,
    ) -> FrameResult:
        if self.enricher is None:
            return frame
        return await self.enricher.aenrich(frame, cols=cols)


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
        watermark_field: str | None = getattr(self, "_watermark_field", None)
        if watermark_field is not None:
            return self._materialize_incremental(
                reload=reload,
                reload_options=reload_options,
                write_index=write_index,
                overwrite=overwrite,
                persist=persist,
                **load_options,
            )
        return self._materialize_full(
            reload=reload,
            reload_options=reload_options,
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
            **load_options,
        )

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
        watermark_field: str | None = getattr(self, "_watermark_field", None)
        if watermark_field is not None:
            return await self._amaterialize_incremental(
                reload=reload,
                reload_options=reload_options,
                write_index=write_index,
                overwrite=overwrite,
                persist=persist,
                **load_options,
            )
        return await self._amaterialize_full(
            reload=reload,
            reload_options=reload_options,
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
            **load_options,
        )

    def _materialize_full(
        self,
        *,
        reload: bool = False,
        reload_options: Mapping[str, Any] | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
        **load_options: Any,
    ) -> ParquetMaterializationResult:
        result = self.write(
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
            **load_options,
        )
        frame = None
        if reload:
            frame = self.from_parquet(**self._reload_options(reload_options))
        return ParquetMaterializationResult(path=result.path, frame=frame)

    async def _amaterialize_full(
        self,
        *,
        reload: bool = False,
        reload_options: Mapping[str, Any] | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
        **load_options: Any,
    ) -> ParquetMaterializationResult:
        result = await self.awrite(
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
            **load_options,
        )
        frame = None
        if reload:
            frame = await self.afrom_parquet(**self._reload_options(reload_options))
        return ParquetMaterializationResult(path=result.path, frame=frame)

    def _update_watermark_from_frame(
        self, frame: Any, watermark_field: str
    ) -> None:
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
        self,
        *,
        reload: bool = False,
        reload_options: Mapping[str, Any] | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
        **load_options: Any,
    ) -> ParquetMaterializationResult:
        watermark_field = self._prepare_incremental_filters(load_options)
        result = self.write(
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
            **load_options,
        )
        frame = None
        if reload:
            if result.files:
                frame = self._finalize_reload_frame(
                    self.from_parquet(**self._reload_options(reload_options)), watermark_field
                )
        else:
            self._update_watermark_from_frame(self.source.load(**load_options), watermark_field)
        return ParquetMaterializationResult(path=result.path, frame=frame)

    async def _amaterialize_incremental(
        self,
        *,
        reload: bool = False,
        reload_options: Mapping[str, Any] | None = None,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
        **load_options: Any,
    ) -> ParquetMaterializationResult:
        watermark_field = self._prepare_incremental_filters(load_options)
        result = await self.awrite(
            write_index=write_index,
            overwrite=overwrite,
            persist=persist,
            **load_options,
        )
        frame = None
        if reload:
            if result.files:
                frame = self._finalize_reload_frame(
                    await self.afrom_parquet(**self._reload_options(reload_options)), watermark_field
                )
        else:
            self._update_watermark_from_frame(await self.source.aload(**load_options), watermark_field)
        return ParquetMaterializationResult(path=result.path, frame=frame)


def advance_watermark_incremental(frame: Any, watermark_field: str) -> Any | None:
    return advance_watermark(frame, watermark_field=watermark_field)


__all__ = ["ParquetMaterializationResult", "ParquetPipeline", "SinkPipeline"]
