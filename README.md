# boti-data

`boti-data` is the **data access and data transformation layer** of the Boti ecosystem.

It builds on top of `boti` and gives teams a reusable interface for working with structured data across databases, parquet datasets, schema-controlled transformations, and distributed or partitioned loading workflows.

## What `boti-data` is for

Many teams have the same recurring problem: business logic depends on data that lives in multiple places, arrives in slightly different shapes, and is loaded through a mix of notebooks, scripts, ad hoc SQL, and one-off helpers.

`boti-data` helps turn that into a more coherent data access layer.

It is designed for codebases that need to:

- connect to named data sources consistently
- reflect or model database tables without hand-writing everything up front
- load data through a gateway instead of bespoke query snippets everywhere
- normalise and validate schemas before downstream use
- combine parquet and database workflows in one library
- scale from simple local reads to partitioned or distributed loading

## Problems `boti-data` solves

`boti-data` is useful when data code is suffering from issues like:

- repeated connection boilerplate across notebooks and services
- slow, fragile query code copied from place to place
- inconsistent schema assumptions between producers and consumers
- difficult transitions from exploratory analysis to reusable pipelines
- manual join and field-mapping logic repeated in many modules
- no common abstraction for loading data from SQL and parquet sources

By centralising those patterns, `boti-data` reduces duplicated plumbing and makes transformations easier to reason about.

## Why `boti-data` can make a huge difference

The biggest benefit of `boti-data` is that it creates a **shared data interface** between infrastructure and business logic.

That means teams can spend less time rewriting access code and more time working on actual transformations, validation rules, and downstream decisions.

It can make a major difference when:

- analysts and engineers share the same source systems
- a notebook prototype needs to become production code
- multiple data products depend on the same tables or parquet layouts
- schema drift is a recurring source of errors
- large extracts need partitioning or distributed execution
- teams want a clean boundary between connection details and transformation logic

## Domain areas where it is especially valuable

`boti-data` is intentionally general-purpose, but it is especially strong in domains where structured operational data must be transformed into reliable analytical or decision-ready datasets.

Examples include:

- **analytics engineering**: building reusable source loaders, schema maps, and standardised transformations
- **business operations**: consolidating data from transactional systems, planning tools, and operational databases
- **finance and controlling**: reconciling structured data with explicit schema expectations and repeatable joins
- **risk, compliance, and audit**: validating input shape, tracing transformations, and standardising access patterns
- **customer and product analytics**: joining behavioural and operational datasets with less custom plumbing
- **supply chain and logistics**: unifying inventory, movement, order, and status data from several systems
- **data platform and internal tooling**: giving teams a common gateway layer instead of ad hoc connectors
- **ML feature preparation**: building reliable dataset assembly steps from SQL and parquet sources

In those settings, the gains are not just convenience. They show up as better reuse, fewer integration bugs, and faster movement from exploration to production.

## Core capabilities

- SQL database resources
- async and sync database access helpers
- SQLAlchemy model reflection and registries
- connection catalogues
- parquet resources and readers
- gateway-style loading APIs
- filter expressions
- schema normalisation and validation helpers
- field mapping and join helpers
- partitioned and distributed data workflows

## Installation

Install directly:

```bash
pip install boti-data
```

Or install through the core package extra:

```bash
pip install "boti[data]"
```

`boti-data` doesn't bundle a database driver by default — SQLAlchemy resolves the driver package from your connection URL's scheme (`mysql+pymysql://`, `postgresql+asyncpg://`, ...), so install the one matching your database:

```bash
pip install "boti-data[mysql]"      # asyncmy + pymysql
pip install "boti-data[postgres]"   # asyncpg + psycopg[binary]
```

If a Postgres table has a [pgvector](https://github.com/pgvector/pgvector) `vector` column, install `boti-data[pgvector]` too — this registers pgvector's `Vector` type with SQLAlchemy's reflection so `SqlAlchemyModelBuilder`/`SqlRepository`/`DataGateway` map that column to a usable `list[float]` (with its dimension preserved) instead of an unreflectable `NullType`:

```bash
pip install "boti-data[pgvector]"
```

The base install is deliberately light: single-row transactional access (`SqlRepository`/`AsyncSqlRepository`, see below) works out of the box, with no dask/pandas/polars pulled in. The dataframe-shaped, bulk-oriented side of the package (`DataGateway`/`DataHelper`, `ParquetPipeline`, `HybridDataset`, the sink pipelines, ...) needs the `dataframes` extra:

```bash
pip install "boti-data[dataframes]"  # dask + pandas + polars + boti-dask
```

Every name importable from `boti_data` is loaded lazily on first access, so `import boti_data` alone never touches dask/pandas/polars — only accessing a dataframe-shaped name does.

## Imports

`boti-data` uses the top-level Python package `boti_data`. Dask runtime/session/resilience helpers live in `boti_dask` and are not re-exported here.

```python
from boti_data import (
    AsyncFrameEnricher,
    AttachmentSpec,
    CsvSink,
    CsvSinkConfig,
    ConnectionCatalog,
    DatacubeConfig,
    DatacubeContract,
    DataGateway,
    DataHelper,
    FieldMap,    
    HybridDataset,
    JsonlSink,
    JsonlSinkConfig,
    ParquetDataConfig,
    ParquetDataResource,
    ParquetPipeline,
    SinkRegistry,
    SinkPipeline,
    available_sinks,
    create_sink,
    register_sink,
    AsyncSqlRepository,
    SqlAlchemyModelBuilder,
    SqlDatabaseConfig,
    SqlDatabaseResource,
    SqlRepository,
)
```

For Dask runtime/session/resilience utilities, import from `boti_dask` directly:

```python
from boti_dask import (
    DaskSession,
    UniqueValuesExtractor,
    apply_recommended_dask_config,
    async_safe_compute,
    async_safe_gather,
    async_safe_head,
    async_safe_persist,
    async_safe_wait,
    dask_is_empty,
    dask_is_probably_empty,
    dask_session,
    inspect_graph,
    safe_compute,
    safe_gather,
    safe_head,
    safe_persist,
    safe_wait,
)
```

Lower-level modules are also available:

```python
from boti_data.db import SqlDatabaseConfig, SqlDatabaseResource
from boti_data.datacube import DatacubeConfig, DatacubeContract
from boti_data.gateway import DataGateway
from boti_data.parquet import ParquetDataConfig, ParquetDataResource
from boti_data.schema import validate_schema
```

## Datacube backend

`DataGateway` also supports a callable-backed `datacube` backend for in-process cube loaders while keeping the same return-type API (`pandas`/`dask`/`arrow`/`polars`).

```python
import pandas as pd

from boti_data import DatacubeConfig, DatacubeContract, DataGateway


def loader(request):
    frame = pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]})
    if request.filters.get("status__exact") == "active":
        frame = frame[frame["status"] == "active"]
    return frame


contract = DatacubeContract(
    request_transformer=lambda request: request.model_copy(update={"cube": request.cube or "orders_v2"}),
    frame_transformer=lambda frame, request: frame.assign(cube=request.cube),
)

with DataGateway(
    DatacubeConfig(loader=loader, default_cube="orders", contract=contract),
    table="orders",
    sticky_filters={"status__exact": "active"},
) as gateway:
    df = gateway.load(return_type="pandas")
```

You can also configure this path with `DataGateway.from_config({"backend": "datacube", ...})` or `DataHelper({"backend": "datacube", ...})`.

See `docs/DATACUBE_CONTRACT.md` for hook ordering, loader precedence, and validator rejection guidance.
See `docs/SECURITY_HARDENING.md` for SQL/raw-sql policy and distributed credential hardening defaults.
For a runnable rejection flow, see `examples/data_facade_datacube_contract_rejection.py`.

## DataHelper

`DataHelper` is the primary entry point for most use cases. It is a thin facade over `DataGateway` that provides a clean, consistent interface for loading data whether you are working locally, in a notebook, or inside a distributed Dask pipeline.

### Creating a DataHelper

`DataHelper` accepts a `DataGateway`, a backend config object, or a plain dict:

```python
from boti_data import DataHelper, SqlDatabaseConfig

# From a config object
config = SqlDatabaseConfig(
    connection_url="mysql+pymysql://user:pass@host/mydb",
    query_only=True,
)
helper = DataHelper(config, table="orders")

# From a dict (useful for config-driven setups)
helper = DataHelper({
    "backend": "sqlalchemy",
    "connection_url": "mysql+pymysql://user:pass@host/mydb",
    "table": "orders",
    "query_only": True,
})

# From keyword arguments
helper = DataHelper(
    backend="sqlalchemy",
    connection_url="mysql+pymysql://user:pass@host/mydb",
    table="orders",
)
```

Using `DataHelper` as a context manager ensures connections are properly closed:

```python
with DataHelper(config, table="orders") as helper:
    df = helper.load(status="confirmed")
```

Async context managers are also supported:

```python
async with DataHelper(config, table="orders") as helper:
    df = await helper.aload(status="confirmed")
```

For synchronous scripts that still need to call async load paths, use the explicit sync bridge:

```python
with DataHelper(config, table="orders") as helper:
    df = helper.aload_sync(status="confirmed")
```

If an event loop is already running (for example notebooks or ASGI handlers), use `await helper.aload(...)` directly.

## ParquetPipeline

`ParquetPipeline` is the first orchestration layer built on top of `DataHelper` and `HybridDataset`.
It materializes a lazy source load into a parquet dataset directory and can optionally reload the
written dataset through `ParquetReader`.

```python
from boti_data import DataHelper, ParquetPipeline

helper = DataHelper(
    backend="sqlalchemy",
    connection_url=f"sqlite:///{db_path}",
    poolclass="sqlalchemy.pool.NullPool",
    query_only=False,
    table="source_events",
)

pipeline = ParquetPipeline(
    helper,
    {
        "backend": "parquet",
        "storage_path": "/tmp/source_events_dataset",
        "partition_on": ["partition_date"],
    },
    date_field="event_date",
)

result = pipeline.materialize(
    filters={"status__exact": "active"},
    reload=True,
    reload_options={"filters": {"partition_date__exact": "2026-04-17"}},
)

assert result.reloaded is True
frame = result.frame
```

Use `materialize()` / `amaterialize()` for the one-step workflow, or call `to_parquet()` and
`from_parquet()` separately when you want explicit control over the write and reload phases.
See `examples/data_parquet_pipeline.py` for a runnable end-to-end example.

## SinkPipeline and Sinks

Phase 3 adds a minimal sink/plugin layer next to `ParquetPipeline`.

- `SinkPipeline` orchestrates `DataHelper` / `HybridDataset` loads into any sink implementing the pipeline sink contract
- `CsvSink` and `JsonlSink` are write-only dataset sinks (`.csv` / `.jsonl` shards)
- `ParquetPipeline` now uses the same sink abstraction internally via `ParquetSink`

Use the sink registry for named sink creation:

```python
from boti_data import SinkPipeline, available_sinks

assert "jsonl" in available_sinks()

pipeline = SinkPipeline(
    helper,
    "jsonl",
    sink_config={
        "storage_path": "/tmp/source_events_jsonl",
        "partition_on": ["partition_date"],
    },
    date_field="event_date",
)
result = pipeline.write(filters={"status__exact": "active"})
```

```python
from boti_data import CsvSink, DataHelper, SinkPipeline

helper = DataHelper(
    backend="sqlalchemy",
    connection_url=f"sqlite:///{db_path}",
    poolclass="sqlalchemy.pool.NullPool",
    query_only=False,
    table="source_events",
)

sink = CsvSink(
    {
        "storage_path": "/tmp/source_events_csv",
        "partition_on": ["partition_date"],
    }
)

pipeline = SinkPipeline(helper, sink, date_field="event_date")
result = pipeline.write(filters={"status__exact": "active"})

assert result.files
```

`CsvSink`/`JsonlSink` are intentionally write-only in this release. Use `ParquetPipeline` when you need
write + reload orchestration. See `examples/data_csv_sink_pipeline.py` and
`examples/data_jsonl_sink_pipeline.py` for runnable examples.

## Enrichment v1

`AsyncFrameEnricher` adds declarative attachment-based enrichment before downstream writes.
Select attachments by `AttachmentSpec.key` via `cols=[...]`, and enforce bounded unique-value
extraction with `max_unique_values`.

```python
from boti_data import AsyncFrameEnricher, AttachmentSpec

enricher = AsyncFrameEnricher(
    [
        AttachmentSpec(
            key="customer_segment",
            required_cols={"customer_id"},
            attachment_fn=customer_segment_attachment,
            col_to_kwarg={"customer_id": "ids"},
            left_on=["customer_id"],
            right_on=["id"],
            drop_cols=["id"],
            max_unique_values=5000,
        )
    ]
)
enriched = await enricher.aenrich(base_frame, cols=["customer_segment"])
```

`SinkPipeline` accepts `enricher=...` and optional `enrich_cols=[...]` to apply enrichment right
before sink writes.

### Raw `sql=` safety policy (eager SQL only)

`DataGateway` supports convenience `sql="..."` reads for eager SQL paths, with explicit safety controls:

- raw SQL is validated as read-only single-statement `SELECT`/`WITH`
- mutating SQL (`INSERT`, `UPDATE`, `DELETE`, `DROP`, etc.) is rejected
- multi-statement SQL is rejected
- lazy SQL still requires `statement` + `model` (raw `sql=` is eager-only)

Use constructor policy `raw_sql_policy` to control availability:

- `raw_sql_policy="readonly_opt_in"` (default): call site must pass `allow_raw_sql=True`
- `raw_sql_policy="disabled"`: raw `sql=` is blocked even if `allow_raw_sql=True`

```python
from boti_data import DataGateway, SqlDatabaseConfig

config = SqlDatabaseConfig(connection_url="sqlite:///example.db", query_only=True)

with DataGateway(config, raw_sql_policy="readonly_opt_in") as gateway:
    df = gateway.load(
        sql="SELECT id, status FROM users WHERE status = :status",
        params={"status": "active"},
        as_pandas=True,
        allow_raw_sql=True,
    )
```

Prefer `statement` + `model` for production query paths whenever possible, especially when you need lazy/partitioned execution.

---

## Output engines: pandas, polars, and dask

`DataHelper` exposes three engine-bound views that pin the output type for a call chain:

```python
helper = DataHelper(config, table="orders")

# Always returns pandas.DataFrame
df = helper.pandas.load(status="confirmed")

# Always returns polars.DataFrame
df = helper.polars.load(status="confirmed")

# Always returns dask.dataframe.DataFrame (lazy)
df = helper.dask.load(status="confirmed")
```

These are the cleanest way to use a single helper across different downstream contexts. You can also pass `return_type` explicitly to `load` or `aload` when you need more control:

```python
# Explicit return_type on a single call
df = helper.load(status="confirmed", return_type="polars")
df = helper.load(status="confirmed", return_type="pandas")
df = helper.load(status="confirmed", return_type="dask")
df = helper.load(status="confirmed", return_type="arrow")  # pyarrow.Table
```

### Choosing an output engine

| Engine | Type returned | Best for |
|---|---|---|
| `pandas` | `pandas.DataFrame` | Small-to-medium data, notebooks, local analysis |
| `polars` | `polars.DataFrame` | CPU-intensive transforms, single-machine performance |
| `arrow` | `pyarrow.Table` | Zero-copy interchange, serialisation, ML pipelines |
| `dask` | `dask.dataframe.DataFrame` | Large datasets, distributed clusters, lazy evaluation |
| `auto` | decided at runtime | Unknown result size; `boti-data` uses backend-aware size heuristics |

`return_type="auto"` uses pandas for small results and switches to Dask for larger scans, with backend-specific checks:

- SQL: probes up to 10,000 rows (and respects `limit` / explicit `partitioned` controls).
- Parquet: uses pandas when the selected scan is small enough to estimate eagerly (typically up to 4 files and <= 32 MB total), otherwise Dask.

Use `auto` when result size is uncertain and you want sensible defaults without hardcoding an engine.

---

## Non-distributed usage

For local analysis, notebooks, or small-scale pipelines, use `DataHelper` without any Dask cluster. The default output is a Dask DataFrame, but you can force pandas or polars.

```python
from boti_data import DataHelper, SqlDatabaseConfig

config = SqlDatabaseConfig(
    connection_url="sqlite:///local.db",
    query_only=True,
)

with DataHelper(config, table="orders") as helper:
    # Pandas — eager, in-memory
    df = helper.pandas.load(status="shipped")

    # Polars — eager, high-performance single-machine
    df = helper.polars.load(status="shipped")

    # Date range load with pandas output
    df = helper.pandas.load_period("created_at", "2024-01-01", "2024-03-31")
```

For async contexts (FastAPI, async services):

```python
async def get_orders(status: str) -> pd.DataFrame:
    async with DataHelper(config, table="orders") as helper:
        return await helper.pandas.aload(status=status)
```

---

## Dask resilience helpers

`boti-data` integrates with `boti_dask` for opt-in Dask runtime and resilience workflows.

Available helpers:

- `inspect_graph(...)`
- `safe_compute(...)`
- `safe_head(...)`
- `safe_gather(...)`
- `safe_persist(...)`
- `safe_wait(...)`
- `async_safe_compute(...)`
- `async_safe_head(...)`
- `async_safe_gather(...)`
- `async_safe_persist(...)`
- `async_safe_wait(...)`
- `dask_is_probably_empty(...)`
- `dask_is_empty(...)`
- `UniqueValuesExtractor`
- `apply_recommended_dask_config(...)`

### Core rules

- Dask DataFrame-like graphs do **not** silently fall back to local threaded compute.
- Retry behavior is limited to recoverable communication failures.
- The helpers are opt-in and designed to complement explicit `DaskSession` ownership.

### Minimal example

```python
import asyncio
import dask
import dask.dataframe as dd
import pandas as pd

from boti_dask import (
    apply_recommended_dask_config,
    async_safe_compute,
    dask_session,
    inspect_graph,
    safe_compute,
    safe_persist,
    safe_wait,
)

frame = dd.from_pandas(pd.DataFrame({"id": [1, 2, 3]}), npartitions=2)

with apply_recommended_dask_config():
    with dask_session(cluster_kwargs={"n_workers": 1, "threads_per_worker": 1, "processes": False}) as client:
        metrics = inspect_graph(frame)
        persisted = safe_persist(frame, dask_client=client)
        safe_wait(persisted, dask_client=client)
        total = safe_compute(frame["id"].sum(), dask_client=client)
        preview = safe_head(frame, n=2, dask_client=client)

        async def run_async() -> int:
            delayed_value = dask.delayed(lambda: 6 * 7)()
            return await async_safe_compute(delayed_value, dask_client=client)

        async_total = asyncio.run(run_async())
```

### Dry-run and preview on gateway loads

Use `dry_run=True` to build and inspect a lazy Dask load graph without materializing it, and `preview(...)` / `apreview(...)` to safely sample rows inside the session:

```python
with DataHelper.session(scheduler_address="tcp://scheduler:8786", verify_connectivity=True) as client:
    with DataHelper(config, table="transactions") as helper:
        ddf = helper.dask.load(year=2024, dry_run=True, diagnostics=True)
        preview = helper.dask.preview(year=2024, n=5)
```

### Resilient joins

For large indexed distributed joins, opt into resilient persistence:

```python
joined = helper.left_join(
    left,
    right,
    join_key="id",
    join_schema_map={"id": "Int64"},
    persist=True,
    resilient=True,
    diagnostics=True,
)
```

### Parquet sources

```python
from boti_data import DataHelper, ParquetDataConfig

config = ParquetDataConfig(
    parquet_storage_path="/data/orders/",
    parquet_start_date=date(2024, 1, 1),
    parquet_end_date=date(2024, 3, 31),
)

with DataHelper(config) as helper:
    df = helper.pandas.load()
    df = helper.polars.load()
```

---

## Distributed usage with Dask

For large datasets or cluster workloads, `DataHelper` integrates natively with Dask. The `DataHelper.session()` factory creates a `DaskSession` that manages cluster and client lifecycle.

### Local cluster (development)

```python
from dask.distributed import LocalCluster
from boti_data import DataHelper, SqlDatabaseConfig

config = SqlDatabaseConfig(
    connection_url="mysql+pymysql://user:pass@host/mydb",
    query_only=True,
    worker_connection_env_var="DB_URL",  # see pickleable section below
)

with DataHelper.session(cluster_factory=LocalCluster) as client:
    with DataHelper(config, table="orders") as helper:
        # Returns dask.dataframe.DataFrame — lazy, partitioned
        ddf = helper.dask.load(status="confirmed")

        # Trigger computation
        df = ddf.compute()
```

### Remote cluster

```python
with DataHelper.session(scheduler_address="tcp://scheduler:8786") as client:
    with DataHelper(config, table="events") as helper:
        ddf = helper.dask.load(region="EU", return_type="dask")
        result = ddf.groupby("customer_id").agg({"amount": "sum"}).compute()
```

### Persisting on the cluster

Use `persist=True` to push the loaded data into distributed memory before further computation. This avoids re-reading from the database on every downstream operation:

```python
with DataHelper.session(scheduler_address="tcp://scheduler:8786") as client:
    with DataHelper(config, table="transactions") as helper:
        # Data is loaded and held in cluster memory
        ddf = helper.load(year=2024, persist=True)

        # Subsequent operations reuse the persisted graph
        monthly = ddf.groupby("month").agg({"amount": "sum"}).compute()
        by_region = ddf.groupby("region").size().compute()

        # Preview persisted data before the managed session closes.
        preview = helper.preview(year=2024, n=5, persist=True)
```

### Shared sessions for repeated notebook cells

When repeated cells need to reuse the same scheduler connection, opt into shared-session reuse explicitly:

```python
with DataHelper.session(
    scheduler_address="tcp://scheduler:8786",
    shared=True,
    shared_key="notebook-main-cluster",
) as client:
    with DataHelper(config, table="transactions") as helper:
        ddf = helper.dask.load(year=2024)
```

### HybridDataset distributed pattern

Use this pattern when you split reads across historical and live sources but want one
distributed execution context. `HybridDataset` composes two helpers, and
`DataHelper.session(...)` keeps scheduler ownership explicit.

```python
import asyncio

from dask.distributed import LocalCluster
from boti_data import DataHelper, HybridDataset

historical = DataHelper(backend="sqlalchemy", connection_url="sqlite:///events.db", table="events_hist")
live = DataHelper(backend="sqlalchemy", connection_url="sqlite:///events.db", table="events_live")
dataset = HybridDataset(historical, live, date_field="event_date", split_date="2026-04-18")

with LocalCluster(n_workers=1, threads_per_worker=1, processes=False, dashboard_address=":0") as cluster:
    with DataHelper.session(
        scheduler_address=cluster.scheduler_address,
        verify_connectivity=True,
        shared=True,
        shared_key="hybrid-dataset-distributed",
    ):
        ddf = dataset.dask.load(start="2026-04-14", end="2026-04-19", diagnostics=True)
        frame = ddf.compute()
        eager_frame = asyncio.run(
            dataset.aload(start="2026-04-16", end="2026-04-18", return_type="pandas")
        )
```

For a complete runnable script, see `examples/data_hybrid_dataset_distributed.py`.

### Semi-join across distributed frames

```python
import pandas as pd

active_customers = pd.Series([1001, 1002, 1003, 1099])

with DataHelper(config, table="orders") as helper:
    # Loads only rows where customer_id is in active_customers
    ddf = helper.semi_join(active_customers, on="customer_id")
    df = ddf.compute()
```

`semi_join` also accepts Dask Series, enabling fully lazy distributed joins:

```python
with DataHelper(config, table="customers") as customer_helper:
    with DataHelper(config, table="orders") as order_helper:
        active_ids = customer_helper.dask.load(active=True)["customer_id"]

        # Lazy — no computation happens yet
        orders_ddf = order_helper.semi_join(active_ids, on="customer_id")

        # Single compute triggers both loads
        result = orders_ddf.compute()
```

---

## The `pickleable` setting in distributed systems

When Dask distributes tasks across workers, it serialises (pickles) the task function and all its arguments to send them over the network. This creates a problem: **database connection objects, engine pools, and credentials cannot be pickled**.

`boti-data` addresses this through the `worker_connection_env_var` setting on `SqlDatabaseConfig`.

### How it works

Instead of serialising the full `SqlDatabaseConfig` (which contains the connection URL and credentials), `boti-data` extracts a minimal `WorkerSqlConfig` for each worker task. If `worker_connection_env_var` is set, workers read the DSN from that environment variable instead of having it embedded in the task payload.

```
Scheduler                              Worker
─────────                              ──────
SqlDatabaseConfig (full config)        WorkerSqlConfig (minimal, safe to pickle)
  connection_url = "mysql://..."   →     connection_env_var = "DB_URL"
  pool_size = 10                         query_only = True
  ...                                    pool_recycle = 1800
                                         (reads DB_URL from os.environ on worker)
```

### Setting it up

**Step 1.** Set the environment variable on all workers. For a local cluster:

```bash
export DB_URL="mysql+pymysql://user:pass@host/mydb"
```

For a Kubernetes-deployed cluster, inject it as a secret.

**Step 2.** Reference the variable in your config:

```python
from boti_data import DataHelper, SqlDatabaseConfig

config = SqlDatabaseConfig(
    connection_url="mysql+pymysql://user:pass@host/mydb",
    query_only=True,
    worker_connection_env_var="DB_URL",  # workers use this instead of pickling credentials
)
```

**Step 3.** Use `DataHelper` normally. Credential serialisation is handled transparently:

```python
with DataHelper.session(scheduler_address="tcp://scheduler:8786") as client:
    with DataHelper(config, table="orders") as helper:
        ddf = helper.dask.load(status="confirmed")
        result = ddf.compute()
```

### Why this matters

Without `worker_connection_env_var`, using a real database DSN with distributed Dask will either:

- fail with a pickle error (connection pool objects are not serialisable)
- embed plaintext credentials in task payloads that flow through scheduler memory and worker logs

Setting `worker_connection_env_var` prevents both problems and is the recommended approach for any distributed SQL workflow.

### Parquet in distributed settings

Parquet resources use `fsspec` for filesystem access. The filesystem object is not pickled directly; instead, `ParquetDataResource` uses a `fs_factory` callable or a `filesystem_profile` name that workers can use to reconstruct the filesystem independently. When `filesystem_profile` is used, filesystem creation is routed through `ConnectionCatalog` adapters so retry/cache behavior stays consistent.

```python
from boti_data import DataHelper, ParquetDataConfig, ConnectionCatalog

catalog = ConnectionCatalog()
catalog.load_filesystem("s3_prod", prefix="S3_")  # reads S3_ENDPOINT, S3_KEY, etc.

config = ParquetDataConfig(
    filesystem_profile="s3_prod",  # workers resolve filesystem from catalog
    parquet_storage_path="s3://my-bucket/orders/",
)

with DataHelper.session(cluster_factory=LocalCluster) as client:
    with DataHelper(config) as helper:
        ddf = helper.dask.load()
        result = ddf.compute()
```

---

## Choosing between distributed and non-distributed

Use the following as a guide:

| Scenario | Recommended approach |
|---|---|
| Exploratory analysis in a notebook | `helper.pandas.load()` — simple, no overhead |
| Single-machine pipeline, large-ish data | `helper.polars.load()` — fast, low memory |
| Result size unknown at design time | `helper.load(return_type="auto")` — adapts |
| Data does not fit in one machine's RAM | `helper.dask.load()` + local or remote cluster |
| Heavy transforms over millions of rows | `helper.dask.load()` + Dask cluster |
| Async service (FastAPI, ASGI) | `await helper.pandas.aload()` or `await helper.dask.aload()` |
| Sync script that needs async load path | `helper.aload_sync(...)` |
| Joining two large tables on a cluster | `helper.semi_join(series, on="key")` |
| Scheduled overnight batch job | Dask cluster + `persist=True` for multi-pass jobs |

**Rule of thumb:** start with `pandas`, switch to `polars` when single-machine performance matters, and move to `dask` when data size exceeds available RAM or when the task benefits from parallelism across workers.

---

## Examples

### Repository (lightest-weight, no dataframe dependencies)

For single-row transactional access — get/insert/update/delete by primary key — without pulling in dask/pandas/polars at all. Rows come back as plain `dict[str, Any]`, never a DataFrame.

`query_only` defaults to `True` on `SqlRepository`/`AsyncSqlRepository` themselves and always wins over whatever the passed-in `config.query_only` says — insert/update/delete raise until you opt in explicitly with `query_only=False`. This is deliberate: a `config` object might be shared with or reused from a context where writes are already enabled, and a repository shouldn't silently inherit write access nobody asked it for.

```python
from boti_data import SqlDatabaseConfig, SqlRepository

config = SqlDatabaseConfig(connection_url="sqlite:///example.db")

with SqlRepository(config, "users", query_only=False) as users:
    row = users.insert({"id": 1, "name": "Ada"})
    row = users.get(1)              # -> dict | None
    row = users.update(1, {"name": "Grace"})
    deleted = users.delete(1)       # -> bool
```

Omit `query_only=False` (or read-only access is all you need) and writes raise instead of silently succeeding:

```python
with SqlRepository(config, "users") as users:
    users.get(1)                    # fine
    users.insert({"id": 2, "name": "Bob"})  # raises SQLAlchemyError
```

An async counterpart, `AsyncSqlRepository`, mirrors the same four methods as coroutines and is used as an async context manager:

```python
from boti_data import AsyncSqlRepository

async with AsyncSqlRepository(config, "users", query_only=False) as users:
    row = await users.get(1)
```

#### Multi-table atomic transactions

By default each `SqlRepository` owns its own connection and commits after every write — fine for independent single-row operations, but not when several writes (potentially across different tables) need to succeed or fail together. Pass an already-open `session` instead of `config` to bind a repository to a transaction you control; writes `flush()` instead of committing, so nothing is durable until you commit the session yourself:

```python
with SqlDatabaseResource(config) as db, db.session() as session:
    with SqlRepository(session=session, table_name="audit_log") as audit_log:
        with SqlRepository(session=session, table_name="embeddings") as embeddings:
            audit_log.insert({"event": "enroll"})
            embeddings.insert({"subject_id": 1, "vector": [...]})
    session.commit()  # both rows commit together, or neither does on an exception
```

`SqlUnitOfWork` wraps this pattern for the common case — one config, several tables, one commit/rollback:

```python
from boti_data import SqlDatabaseConfig, SqlUnitOfWork

config = SqlDatabaseConfig(connection_url="postgresql+psycopg://user:pass@host/mydb")

with SqlUnitOfWork(config, query_only=False) as uow:
    uow.repository("audit_log").insert({"event": "enroll"})
    uow.repository("embeddings").insert({"subject_id": 1, "vector": [...]})
# commits both on clean exit; rolls back both on any exception
```

`query_only` on a `session`-bound `SqlRepository` is inert — the session's own class (e.g. a `ReadOnlySession`) already determines writability, since this path never builds a new guarded session itself. `AsyncSqlUnitOfWork` mirrors `SqlUnitOfWork`, with an async `repository()` method (returns an already-set-up `AsyncSqlRepository`, so callers don't need to `async with` each one individually):

```python
from boti_data import AsyncSqlUnitOfWork

async with AsyncSqlUnitOfWork(config, query_only=False) as uow:
    audit_log = await uow.repository("audit_log")
    embeddings = await uow.repository("embeddings")
    await audit_log.insert({"event": "enroll"})
    await embeddings.insert({"subject_id": 1, "vector": [...]})
```

### Nearest-neighbour search (pgvector)

`boti_data.db.nearest_neighbors`/`vector_distance` build an `ORDER BY <distance> LIMIT k` statement against a pgvector column, with the query vector bound as a driver parameter — never string-interpolated into the SQL, regardless of how large the vector is. Requires `boti-data[pgvector]` (see [Installation](#installation)) and a table reflected via `SqlAlchemyModelBuilder`/`SqlRepository`/`DataGateway`.

```python
from boti_data import DataGateway, SqlDatabaseConfig
from boti_data.db import SqlAlchemyModelBuilder, nearest_neighbors

config = SqlDatabaseConfig(connection_url="postgresql+psycopg://user:pass@host/mydb", query_only=True)
gateway = DataGateway(config)

model = SqlAlchemyModelBuilder(gateway.resource.engine, "embeddings").build_model()
query_vector = [...]  # e.g. a 512-float face embedding

stmt = nearest_neighbors(model, model.vector, query_vector, k=10, metric="cosine")
matches = gateway.load(statement=stmt, model=model, as_pandas=True)
```

Supported `metric` values map directly to pgvector's operators: `"cosine"` (`<=>`), `"l2"` (`<->`), `"inner_product"` (`<#>`), `"l1"` (`<+>`). Add ordinary filtering with `.where(...)` on the statement before passing it to `gateway.load(...)`.

### SQL resource (low-level)

```python
from boti_data import SqlDatabaseConfig, SqlDatabaseResource

config = SqlDatabaseConfig(connection_url="sqlite:///example.db", query_only=True)

with SqlDatabaseResource(config) as db:
    with db.session() as session:
        rows = session.execute(...)
```

### Gateway (mid-level)

```python
from boti_data import DataGateway, SqlDatabaseConfig

gateway = DataGateway(
    backend="sqlalchemy",
    config=SqlDatabaseConfig(connection_url="sqlite:///example.db", query_only=True),
)
```

### DataHelper — local pandas

```python
from boti_data import DataHelper, SqlDatabaseConfig

config = SqlDatabaseConfig(
    connection_url="postgresql+asyncpg://user:pass@host/mydb",
    query_only=True,
)

with DataHelper(config, table="sales") as helper:
    df = helper.pandas.load(year=2024, region="EMEA")
    print(df.head())
```

### DataHelper — local polars

```python
with DataHelper(config, table="sales") as helper:
    df = helper.polars.load(year=2024)
    summary = df.group_by("region").agg(pl.col("amount").sum())
```

### DataHelper — lazy Dask, no cluster

```python
with DataHelper(config, table="sales") as helper:
    ddf = helper.dask.load(year=2024)
    # Graph is not executed yet; chain transforms lazily
    result = ddf.groupby("region")["amount"].sum().compute()
```

### DataHelper — distributed Dask cluster

```python
from dask.distributed import LocalCluster
from boti_data import DataHelper, SqlDatabaseConfig

config = SqlDatabaseConfig(
    connection_url="mysql+pymysql://user:pass@host/mydb",
    query_only=True,
    worker_connection_env_var="DB_URL",
)

with DataHelper.session(cluster_factory=LocalCluster, n_workers=4) as client:
    with DataHelper(config, table="events") as helper:
        ddf = helper.dask.load(event_type="purchase", persist=True)
        result = ddf.groupby("user_id").size().compute()
```

### DataHelper — async service

```python
from boti_data import DataHelper, SqlDatabaseConfig

config = SqlDatabaseConfig(
    connection_url="mysql+asyncmy://user:pass@host/mydb",
    query_only=True,
)

async def load_orders(status: str) -> pd.DataFrame:
    async with DataHelper(config, table="orders") as helper:
        return await helper.pandas.aload(status=status)
```

### DataHelper — date-range load

```python
with DataHelper(config, table="transactions") as helper:
    # Inclusive date range; dt_field is the semantic field name
    df = helper.pandas.load_period("created_at", "2024-01-01", "2024-06-30")
```

### HybridDataset — historical + live composition

```python
historical = DataHelper(backend="parquet", storage_path="/data/historical", parquet_filename="orders")
live = DataHelper(backend="sqlalchemy", connection_url="sqlite:///live.db", table="orders_live")

dataset = HybridDataset(
    historical,
    live,
    date_field="order_date",
    split_date="2026-04-18",
)

# mixed window (auto -> dask for hybrid composition)
ddf = dataset.load(start="2026-04-15", end="2026-04-20", return_type="auto")

# explicit source override through engine-bound view
pdf = dataset.pandas.load(start="2026-04-18", end="2026-04-20", source="live")
```

### DataHelper — parquet source

```python
from boti_data import DataHelper, ParquetDataConfig
from datetime import date

config = ParquetDataConfig(
    parquet_storage_path="/data/warehouse/orders/",
    parquet_start_date=date(2024, 1, 1),
    parquet_end_date=date(2024, 6, 30),
)

with DataHelper(config) as helper:
    df = helper.pandas.load()
    ddf = helper.dask.load()  # lazy, partitioned read
```

### Connection catalog

```python
from boti_data import ConnectionCatalog, DataHelper

catalog = ConnectionCatalog()
catalog.load_sql("prod", prefix="PROD_DB_")  # reads PROD_DB_URL, PROD_DB_POOL_SIZE, etc.
catalog.load_sql("reporting", prefix="REPORT_DB_")

prod_config = catalog.sql_config("prod")
report_config = catalog.sql_config("reporting")

with DataHelper(prod_config, table="orders") as helper:
    df = helper.pandas.load(status="confirmed")
```

---

## Relationship to `boti`

`boti-data` depends on `boti`, and reuses:

- logging
- resource lifecycle
- secure I/O helpers
- project/environment utilities

If you only need the runtime primitives, install `boti`.
If you need a stronger data access and transformation layer, install `boti-data` or `boti[data]`.

## Development & Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for publishing instructions.
