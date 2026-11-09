"""R8: metrics and tracing are emitted through one configurable provider."""

from __future__ import annotations

import sqlite3

import pytest

from boti_data import DataGateway
from boti_data import telemetry as tm
from boti_data.db.sql_config import SqlDatabaseConfig


@pytest.fixture
def orders_db(tmp_path) -> str:
    path = tmp_path / "orders.sqlite"
    con = sqlite3.connect(path)
    con.execute("create table orders (id integer primary key, total real)")
    con.executemany("insert into orders values (?, ?)", ((i, float(i)) for i in range(1, 501)))
    con.commit()
    con.close()
    return str(path)


@pytest.fixture
def memory_provider():
    provider = tm.configure(tm.InMemoryProvider())
    try:
        yield provider
    finally:
        tm.configure(None)


def _load_pandas(db: str, **options):
    gateway = DataGateway(
        SqlDatabaseConfig(connection_url=f"sqlite:///{db}", query_only=False), table="orders"
    )
    with gateway:
        return gateway.load(
            as_pandas=True, return_type="pandas", partitioned=True, chunk_size=100, **options
        )


def test_default_provider_is_noop_and_env_selects_one(monkeypatch) -> None:
    tm.configure(None)
    assert isinstance(tm.current_provider(), tm.NoopProvider)
    tm._registry.provider = None
    monkeypatch.setenv(tm.ENV_VAR, "memory")
    try:
        assert isinstance(tm.current_provider(), tm.InMemoryProvider)
    finally:
        tm.configure(None)
    with pytest.raises(ValueError, match="Unknown telemetry provider"):
        tm.provider_from_name("carrier-pigeon")


def test_sql_load_records_span_pool_checkouts_and_gate_wait(orders_db, memory_provider) -> None:
    frame = _load_pandas(orders_db)
    assert len(frame) == 500

    loads = memory_provider.spans_named("boti_data.load")
    assert loads and loads[-1].attributes["backend"] == "sqlalchemy"
    assert loads[-1].attributes["return_type"] == "pandas"
    assert loads[-1].attributes["rows"] == 500
    assert loads[-1].error is None

    fetches = memory_provider.spans_named("boti_data.partition.fetch")
    assert fetches and sum(s.attributes["rows"] for s in fetches) == 500
    assert all(s.attributes["gate"].startswith("boti-data:fetch-gate:") for s in fetches)
    assert all("sqlite" not in s.attributes["gate"] for s in fetches)

    assert memory_provider.total("boti_data.pool.checkouts") >= 1
    checkouts = memory_provider.instruments["boti_data.pool.checkouts"].observations
    assert all(len(attrs["engine"]) == 16 for _, attrs in checkouts)
    assert memory_provider.last("boti_data.pool.checked_out") == 0

    waits = memory_provider.instruments["boti_data.gate.wait_seconds"].observations
    assert len(waits) == len(fetches)
    assert all(value >= 0 for value, _ in waits)


def test_failed_load_marks_the_span(orders_db, memory_provider) -> None:
    gateway = DataGateway(
        SqlDatabaseConfig(connection_url=f"sqlite:///{orders_db}", query_only=False), table="orders"
    )
    with gateway, pytest.raises(Exception):
        gateway.load(as_pandas=True, return_type="pandas", sql="select * from nowhere")
    assert memory_provider.spans_named("boti_data.load")[-1].error is not None


def test_opentelemetry_provider_exports_spans_and_metrics(orders_db) -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    provider = tm.OpenTelemetryProvider(
        tracer=tracer_provider.get_tracer("test"), meter=meter_provider.get_meter("test")
    )
    tm.configure(provider)
    try:
        _load_pandas(orders_db)
    finally:
        tm.configure(None)

    names = {span.name for span in exporter.get_finished_spans()}
    assert {"boti_data.load", "boti_data.partition.fetch", "boti_data.partitioned.load"} <= names
    data = reader.get_metrics_data()
    metric_names = {
        metric.name
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert {
        "boti_data.pool.checkouts",
        "boti_data.pool.checked_out",
        "boti_data.gate.wait_seconds",
    } <= metric_names


def test_otel_provider_names_the_extra_when_missing(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _block(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)
    with pytest.raises(ModuleNotFoundError, match=r"boti-data\[otel\]"):
        tm.OpenTelemetryProvider()
