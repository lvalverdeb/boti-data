from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Union, cast

from boti.core.lifecycle import LifecycleCore
from boti.core.lifecycle_pickle import PicklableLifecycleCoreMixin

from boti_data.dataset import HybridDataset
from boti_data.helper import DataHelper
from boti_data.pipelines.registry import create_sink
from boti_data.pipelines.sinks import (
    FrameResult,
    ParquetDestination,
    PipelineSink,
    SinkWriteResult,
)

type PipelineSource = Union[DataHelper, HybridDataset]
type PipelineDestination = ParquetDestination


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

    # Not a copy-pasted twin: option prep is already extracted
    # (_prepare_write_options()/_materialization_load_options()); the
    # remaining body is 3 calls to already-split twin methods (load/aload,
    # _maybe_enrich_sync/_maybe_enrich_async, sink.write/awrite) — duplication
    # is a byproduct of composing other twins, not new copy-paste.
    # spaghetti-ignore[sync-async-duplication]: see above
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

    # Not a copy-pasted twin: thin guard-and-delegate to the injected
    # self.enricher's own enrich()/aenrich() (a Protocol, already marked as a
    # false positive at its own definition) — nothing to unify here either.
    # spaghetti-ignore[sync-async-duplication]: see above
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


__all__ = ["SinkPipeline"]
