# Metrics and tracing

boti-data emits spans and metrics through one process-wide provider, chosen
by a single setting. By default nothing is recorded and the cost is one
attribute lookup per instrumented call. The boti `Logger` stays the log sink;
telemetry adds signals alongside the existing `diagnostics=True` log lines,
it does not replace them.

## Choosing a provider

```python
from boti_data import telemetry

telemetry.configure(telemetry.OpenTelemetryProvider())   # needs boti-data[otel]
telemetry.configure(telemetry.InMemoryProvider())        # tests and diagnostics
telemetry.configure(None)                                # back to the no-op
```

Or from the environment, read on first use: `BOTI_DATA_TELEMETRY=otel`
(`none` is the default). `telemetry.current_provider()` returns the active one.

`OpenTelemetryProvider` uses the tracer and meter providers the application
configured globally and never installs exporters itself, so one OpenTelemetry
setup serves boti-data and boti-dask alike.

## What is emitted

| Instrument | Name | Attributes |
| --- | --- | --- |
| span | `boti_data.load` | `backend`, `return_type`, `execution_mode`, `rows` when materialised |
| span | `boti_data.partitioned.load` | `gate` |
| span | `boti_data.partition.fetch` | `gate`, `partition` when known, `rows` |
| histogram | `boti_data.gate.wait_seconds` | `gate` |
| counter | `boti_data.pool.checkouts` | `engine` |
| gauge | `boti_data.pool.checked_out` | `engine` |

`gate` is the hash of the engine identity that names the cluster-wide fetch
gate; `engine` is the sixteen-character digest of the connection URL. Neither
ever carries the DSN or its credentials.

Pool instruments come from SQLAlchemy's `PoolEvents` on every engine boti-data
builds: the resource engines behind `SqlDatabaseResource`, the async engines
(through `sync_engine`) and the worker-local engines partition fetches use.
The gate histogram records every wait for a fetch slot, the same observations
that feed `fetch_gate_stats()`.

## Prometheus

Point the OpenTelemetry SDK at a Prometheus reader once at service start-up;
boti-data then reports through it without further configuration:

```python
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import start_http_server

from boti_data import telemetry

start_http_server(9464)
metrics.set_meter_provider(MeterProvider(metric_readers=[PrometheusMetricReader()]))
telemetry.configure(telemetry.OpenTelemetryProvider())
```

A scrape of `:9464/metrics` then shows `boti_data_pool_checked_out`,
`boti_data_pool_checkouts_total` and `boti_data_gate_wait_seconds_bucket`,
which is what to alert on for pool exhaustion and gate contention. Traces go
wherever the global tracer provider exports them (an OTLP exporter, for
instance); install it the same way before configuring the provider.

## Tests

`InMemoryProvider` keeps every span and observation in memory, with
`spans_named()`, `total()` and `last()` helpers, so a test can assert on
exactly what a load emitted without any SDK installed.
