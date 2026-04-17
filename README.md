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

## Imports

> **Breaking change:** Dask runtime/session/resilience helpers are no longer re-exported from `boti_data`.
> Update imports like `from boti_data import DaskSession, safe_compute` to `from boti_dask import DaskSession, safe_compute`.

`boti-data` uses the top-level Python package `boti_data`:

```python
from boti_data import (
    ConnectionCatalog,
    DataGateway,
    DataHelper,
    FieldMap,
    ParquetDataConfig,
    ParquetDataResource,
    SqlAlchemyModelBuilder,
    SqlDatabaseConfig,
    SqlDatabaseResource,
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
from boti_data.gateway import DataGateway
from boti_data.parquet import ParquetDataConfig, ParquetDataResource
from boti_data.schema import validate_schema
```

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

Parquet resources use `fsspec` for filesystem access. The filesystem object is not pickled directly; instead, `ParquetDataResource` uses a `fs_factory` callable or a `filesystem_profile` name that workers can use to reconstruct the filesystem independently.

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
| Joining two large tables on a cluster | `helper.semi_join(series, on="key")` |
| Scheduled overnight batch job | Dask cluster + `persist=True` for multi-pass jobs |

**Rule of thumb:** start with `pandas`, switch to `polars` when single-machine performance matters, and move to `dask` when data size exceeds available RAM or when the task benefits from parallelism across workers.

---

## Examples

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
