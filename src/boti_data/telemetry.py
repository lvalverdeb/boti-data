"""Metrics and tracing for boti-data loads, opt-in through one setting.

A :class:`TelemetryProvider` offers four instruments: ``span`` (a context
manager yielding a handle whose ``set`` adds attributes), ``counter``,
``gauge`` and ``histogram``. Three providers ship: :class:`NoopProvider`
(the default; costs one attribute lookup), :class:`InMemoryProvider`
(records everything; for tests and diagnostics) and
:class:`OpenTelemetryProvider` (behind the ``otel`` extra; uses whatever
tracer and meter providers the application configured globally, so one
OpenTelemetry setup serves boti-data and boti-dask alike).

One provider is active per process. Install it with :func:`configure` or
the environment variable ``BOTI_DATA_TELEMETRY`` (``otel`` or ``none``).
The boti ``Logger`` remains the log sink; telemetry adds signals, it does
not replace log lines.

Instruments emitted by the library:

* span ``boti_data.load``: ``backend``, ``return_type``, ``execution_mode``,
  ``rows`` when the result is materialised.
* span ``boti_data.partition.fetch``: ``gate`` (hash of the engine identity),
  ``partition`` index when known, ``rows``.
* histogram ``boti_data.gate.wait_seconds`` (``gate``): time spent waiting for
  a fetch slot, from the same path that feeds ``GateWaitStats``.
* counter ``boti_data.pool.checkouts`` and gauge ``boti_data.pool.checked_out``
  (``engine`` = digest of the connection URL, never the DSN): SQLAlchemy
  pool activity on every engine boti-data builds.
"""

from __future__ import annotations

import contextlib
import functools
import os
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ENV_VAR",
    "InMemoryProvider",
    "NoopProvider",
    "OpenTelemetryProvider",
    "SpanHandle",
    "TelemetryProvider",
    "configure",
    "current_provider",
    "provider_from_name",
]

ENV_VAR = "BOTI_DATA_TELEMETRY"
_INSTRUMENTATION_NAME = "boti_data"


@runtime_checkable
class SpanHandle(Protocol):
    def set(self, key: str, value: Any) -> None: ...


@runtime_checkable
class Counter(Protocol):
    def add(self, value: int = 1, **attributes: Any) -> None: ...


@runtime_checkable
class Gauge(Protocol):
    def set(self, value: float, **attributes: Any) -> None: ...


@runtime_checkable
class Histogram(Protocol):
    def record(self, value: float, **attributes: Any) -> None: ...


@runtime_checkable
class TelemetryProvider(Protocol):
    def span(self, name: str, **attributes: Any) -> AbstractContextManager[SpanHandle]: ...

    def counter(self, name: str) -> Counter: ...

    def gauge(self, name: str) -> Gauge: ...

    def histogram(self, name: str) -> Histogram: ...


# ---------------------------------------------------------------------------
# No-op
# ---------------------------------------------------------------------------


class _NoopInstrument:
    def set(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record(self, *args: Any, **kwargs: Any) -> None:
        return None


_NOOP = _NoopInstrument()


class NoopProvider:
    """Records nothing; the default."""

    # spaghetti-ignore[pass-through-method]: a no-op span is a null context by definition
    def span(self, name: str, **attributes: Any) -> AbstractContextManager[SpanHandle]:
        return contextlib.nullcontext(_NOOP)

    def counter(self, name: str) -> Counter:
        return _NOOP

    def gauge(self, name: str) -> Gauge:
        return _NOOP

    def histogram(self, name: str) -> Histogram:
        return _NOOP


# ---------------------------------------------------------------------------
# In memory
# ---------------------------------------------------------------------------


@dataclass
class SpanRecord:
    name: str
    attributes: dict[str, Any]
    error: str | None = None

    def set(self, key: str, value: Any) -> None:
        self.attributes[key] = value


@dataclass
class _MemoryInstrument:
    """One named instrument; every observation lands in ``observations``."""

    name: str
    observations: list[tuple[float, dict[str, Any]]]
    lock: threading.Lock

    def add(self, value: int = 1, **attributes: Any) -> None:
        with self.lock:
            self.observations.append((value, attributes))

    def set(self, value: float, **attributes: Any) -> None:
        with self.lock:
            self.observations.append((value, attributes))

    def record(self, value: float, **attributes: Any) -> None:
        with self.lock:
            self.observations.append((value, attributes))


@dataclass
class InMemoryProvider:
    """Keeps spans and instrument observations in memory; thread-safe."""

    spans: list[SpanRecord] = field(default_factory=list)
    instruments: dict[str, _MemoryInstrument] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @contextlib.contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[SpanRecord]:
        record = SpanRecord(name=name, attributes=dict(attributes))
        try:
            yield record
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            with self._lock:
                self.spans.append(record)

    def _instrument(self, name: str) -> _MemoryInstrument:
        with self._lock:
            instrument = self.instruments.get(name)
            if instrument is None:
                instrument = _MemoryInstrument(name, [], threading.Lock())
                self.instruments[name] = instrument
            return instrument

    # One in-memory instrument serves every kind; the provider methods are aliases.
    counter = _instrument
    gauge = _instrument
    histogram = _instrument

    def total(self, name: str) -> float:
        """Sum of a counter's or histogram's observations (0 when never used)."""
        instrument = self.instruments.get(name)
        return sum(value for value, _ in instrument.observations) if instrument else 0

    def last(self, name: str) -> float | None:
        """A gauge's most recent value, or ``None`` when never set."""
        instrument = self.instruments.get(name)
        return instrument.observations[-1][0] if instrument and instrument.observations else None

    def spans_named(self, name: str) -> list[SpanRecord]:
        return [span for span in self.spans if span.name == name]

    def reset(self) -> None:
        with self._lock:
            self.spans.clear()
            self.instruments.clear()


# ---------------------------------------------------------------------------
# OpenTelemetry
# ---------------------------------------------------------------------------


class _OtelSpan:
    """Adapts the ``SpanHandle`` shape to an OpenTelemetry span."""

    def __init__(self, span: Any) -> None:
        self._span = span

    # spaghetti-ignore[pass-through-method]: adapts set(key, value) to set_attribute
    def set(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)


class _OtelInstrument:
    """Adapts ``**attributes`` keyword attributes to OpenTelemetry's ``attributes=``."""

    def __init__(self, instrument: Any) -> None:
        self._instrument = instrument

    # spaghetti-ignore[pass-through-method]: keyword attributes become the attributes= mapping
    def add(self, value: int = 1, **attributes: Any) -> None:
        self._instrument.add(value, attributes=attributes)

    # spaghetti-ignore[pass-through-method]: keyword attributes become the attributes= mapping
    def set(self, value: float, **attributes: Any) -> None:
        self._instrument.set(value, attributes=attributes)

    # spaghetti-ignore[pass-through-method]: keyword attributes become the attributes= mapping
    def record(self, value: float, **attributes: Any) -> None:
        self._instrument.record(value, attributes=attributes)


class OpenTelemetryProvider:
    """Spans through ``opentelemetry.trace`` and instruments through ``opentelemetry.metrics``.

    Uses the tracer and meter providers the application configured globally;
    boti-data never installs exporters itself. Needs the ``otel`` extra.
    """

    def __init__(self, *, tracer: Any | None = None, meter: Any | None = None) -> None:
        try:
            from opentelemetry import metrics, trace
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OpenTelemetryProvider needs boti-data's 'otel' extra: "
                "\"pip install 'boti-data[otel]'\" (or \"uv add 'boti-data[otel]'\").",
                name=exc.name,
                path=exc.path,
            ) from exc
        self._tracer = tracer or trace.get_tracer(_INSTRUMENTATION_NAME)
        self._meter = meter or metrics.get_meter(_INSTRUMENTATION_NAME)
        self._instruments: dict[tuple[str, str], _OtelInstrument] = {}
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[_OtelSpan]:
        with self._tracer.start_as_current_span(name, attributes=attributes) as span:
            yield _OtelSpan(span)

    def _get(self, kind: str, name: str) -> _OtelInstrument:
        with self._lock:
            instrument = self._instruments.get((kind, name))
            if instrument is None:
                factory = getattr(self._meter, f"create_{kind}")
                instrument = _OtelInstrument(factory(name))
                self._instruments[(kind, name)] = instrument
            return instrument

    counter = functools.partialmethod(_get, "counter")
    gauge = functools.partialmethod(_get, "gauge")
    histogram = functools.partialmethod(_get, "histogram")


# ---------------------------------------------------------------------------
# Process-wide registry
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type] = {
    "none": NoopProvider,
    "memory": InMemoryProvider,
    "otel": OpenTelemetryProvider,
}


def provider_from_name(name: str) -> Any:
    """Build a provider from its short name: ``none``, ``memory`` or ``otel``."""
    factory = _PROVIDERS.get(name.strip().lower())
    if factory is None:
        raise ValueError(
            f"Unknown telemetry provider {name!r}. Known providers: {', '.join(_PROVIDERS)}."
        )
    return factory()


@dataclass
class _Registry:
    provider: Any | None = None


_registry = _Registry()
_registry_lock = threading.Lock()


def configure(provider: Any | None) -> Any:
    """Install *provider* process-wide (``None`` restores the no-op) and return it."""
    with _registry_lock:
        _registry.provider = NoopProvider() if provider is None else provider
        return _registry.provider


def current_provider() -> Any:
    """The active provider; on first use honours ``BOTI_DATA_TELEMETRY``."""
    provider = _registry.provider
    if provider is None:
        with _registry_lock:
            if _registry.provider is None:
                _registry.provider = provider_from_name(os.environ.get(ENV_VAR, "none"))
            provider = _registry.provider
    return provider
