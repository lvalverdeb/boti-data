from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Union

from boti_data.pipelines.sinks import CsvSink, JsonlSink, ParquetSink, PipelineSink

SinkFactory = Callable[[Union[Mapping[str, Any], Any]], PipelineSink]


class SinkRegistry:
    """Registry for named pipeline sink factories."""

    def __init__(self) -> None:
        self._factories: dict[str, SinkFactory] = {}

    def register(self, name: str, factory: SinkFactory, *, overwrite: bool = False) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Sink name must be a non-empty string.")
        if normalized in self._factories and not overwrite:
            raise ValueError(
                f"Sink '{normalized}' is already registered. Pass overwrite=True to replace it."
            )
        self._factories[normalized] = factory

    def create(self, name: str, config: Mapping[str, Any] | Any) -> PipelineSink:
        normalized = name.strip().lower()
        factory = self._factories.get(normalized)
        if factory is None:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"Unknown sink '{normalized}'. Available sinks: {available}")
        return factory(config)

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def _parquet_factory(config: Mapping[str, Any] | Any) -> PipelineSink:
    return ParquetSink(config)


def _csv_factory(config: Mapping[str, Any] | Any) -> PipelineSink:
    return CsvSink(config)


def _jsonl_factory(config: Mapping[str, Any] | Any) -> PipelineSink:
    return JsonlSink(config)


_DEFAULT_SINK_REGISTRY = SinkRegistry()
_DEFAULT_SINK_REGISTRY.register("parquet", _parquet_factory)
_DEFAULT_SINK_REGISTRY.register("csv", _csv_factory)
_DEFAULT_SINK_REGISTRY.register("jsonl", _jsonl_factory)


def register_sink(name: str, factory: SinkFactory, *, overwrite: bool = False) -> None:
    """Register a sink factory in the global sink registry."""
    _DEFAULT_SINK_REGISTRY.register(name, factory, overwrite=overwrite)


def create_sink(name: str, config: Mapping[str, Any] | Any) -> PipelineSink:
    """Create a sink from the global sink registry."""
    return _DEFAULT_SINK_REGISTRY.create(name, config)


def available_sinks() -> tuple[str, ...]:
    """Return registered sink names from the global sink registry."""
    return _DEFAULT_SINK_REGISTRY.available()


__all__ = [
    "SinkFactory",
    "SinkRegistry",
    "available_sinks",
    "create_sink",
    "register_sink",
]
