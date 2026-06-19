from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, Union, cast

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


class SinkPipeline:
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

    def __enter__(self) -> SinkPipeline:
        self.source.__enter__()
        sink_enter = getattr(self.sink, "__enter__", None)
        if callable(sink_enter):
            sink_enter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        sink_exit = getattr(self.sink, "__exit__", None)
        if callable(sink_exit):
            sink_exit(exc_type, exc_val, exc_tb)
        self.source.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> SinkPipeline:
        await self.source.__aenter__()
        sink_aenter = getattr(self.sink, "__aenter__", None)
        if callable(sink_aenter):
            maybe = sink_aenter()
            if inspect.isawaitable(maybe):
                await cast(Any, maybe)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        sink_aexit = getattr(self.sink, "__aexit__", None)
        if callable(sink_aexit):
            maybe = sink_aexit(exc_type, exc_val, exc_tb)
            if inspect.isawaitable(maybe):
                await cast(Any, maybe)
        await self.source.__aexit__(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        try:
            sink_close = getattr(self.sink, "close", None)
            if callable(sink_close):
                sink_close()
        finally:
            self.source.close()

    async def aclose(self) -> None:
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

    def write(
        self,
        *,
        write_index: bool = False,
        overwrite: bool = True,
        persist: bool = False,
        **load_options: Any,
    ) -> SinkWriteResult:
        resolved_options = dict(load_options)
        enrich_cols = resolved_options.pop("enrich_cols", None)
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
        resolved_options = dict(load_options)
        enrich_cols = resolved_options.pop("enrich_cols", None)
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


__all__ = ["ParquetMaterializationResult", "ParquetPipeline", "SinkPipeline"]
