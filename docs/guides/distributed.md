# Distributed loading

## Distributed usage with Dask

For large datasets or cluster workloads, `DataHelper` integrates natively with Dask. The `DataHelper.session()` factory creates a `DaskSession` that manages cluster and client lifecycle.

### Local cluster (development)

```python
from dask.distributed import LocalCluster
from boti_data import DataHelper, SqlDatabaseConfig

config = SqlDatabaseConfig(
    connection_url="mysql+pymysql://user:pass@host/mydb",
    query_only=True,
    worker_connection_env_var="DB_URL",  # see the distributed guide
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
        sample = helper.load(year=2024, limit=5, persist=True).compute()
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
- Retry behaviour is limited to recoverable communication failures.
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

Use `dry_run=True` to build and inspect a lazy Dask load graph without materializing it, and `limit=` to safely sample rows inside the session:

```python
with DataHelper.session(scheduler_address="tcp://scheduler:8786", verify_connectivity=True) as client:
    with DataHelper(config, table="transactions") as helper:
        ddf = helper.dask.load(year=2024, dry_run=True, diagnostics=True)
        sample = helper.dask.load(year=2024, limit=5).compute()
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

## Sizing connections and fetch concurrency

More database connections do not mean more throughput. Past a modest number they
mean *less*, because connections compete for the same cores, disks and locks. Both
the [PostgreSQL wiki](https://wiki.postgresql.org/wiki/Number_Of_Database_Connections)
and the [HikariCP pool-sizing guide](https://github.com/brettwooldridge/HikariCP/wiki/About-Pool-Sizing)
give the same starting formula:

```
connections = (core_count * 2) + effective_spindle_count
```

Count physical cores, not hyperthreads, and take `effective_spindle_count` as zero
when the working set is cached in RAM. For a 4-core server with one disk that is
about 10 connections in total — not per client, not per worker, **in total**.

The characteristic failure of getting this wrong is counter-intuitive: the database
sits nearly idle, at a few percent CPU, while clients time out. The connections are
all queued rather than working. Raising the pool size makes it worse.

`boti-data` exposes three separate numbers that multiply into that total. Size the
product, not the individual knobs.

### 1. Client-side pool: `pool_size + max_overflow`

`SqlDatabaseConfig` defaults to `pool_size=5` and `max_overflow=10`, so a single
engine bursts to **15** connections, not 5. `max_overflow` is three times
`pool_size`, so the steady-state number is a third of the real ceiling.

```python
config = SqlDatabaseConfig(
    connection_url="postgresql+psycopg://...",
    pool_size=5,       # steady state
    max_overflow=10,   # burst headroom — the ceiling is 5 + 10 = 15
)
```

Engines are cached per distinct configuration, so two differently-configured helpers
against the same database hold two pools: 30 connections, not 15.

### 2. Distributed fetch concurrency: `n_workers × max_concurrent_fetches`

`max_concurrent_fetches` (default `4`, capped at `16`) bounds concurrent partition
fetches **across the whole cluster**: the gate is a `boti_dask.cluster_semaphore`
held on the Dask scheduler and shared by every worker, keyed on the database alone.
Twenty workers at `max_concurrent_fetches=4` open at most 4 fetch connections to that
database, so the number means what it reads.

Two things keep it honest:

- **One limit per database.** Two loads against the same database with different
  limits share one gate; the first limit seen wins and a warning is logged. Limits
  never add up.
- **No cluster, no scheduler gate.** In a local threads run, or with `boti-dask`
  older than 1.1.0, the same gate bounds the **process** instead, so a run with
  several processes against one database multiplies: `n_processes × max_concurrent_fetches`.

With the 10-connection budget above, leave headroom for the client-side pool and set
`max_concurrent_fetches` to the remainder, whatever the worker count.

### 3. Worker connections are unpooled by design

Worker engines always use `NullPool`. Connection pools cannot be pickled and would
bind to the wrong event loop, so each partition fetch opens a fresh connection and
closes it. That is correct for distribution, but it means every fetch pays a full TCP
handshake, TLS negotiation and authentication — on a remote database, tens of
milliseconds each, before a single row is read. A 200-partition load pays it 200 times.

The fix is not to pool inside the workers, but to put a pooler in front of the
database, so that warm server-side connections are reused while workers keep opening
cheap local ones. **The pooling mode matters**, because it interacts with `query_only`:

| Pooler mode | Connection reuse | `query_only=True` |
|---|---|---|
| Session (e.g. PgBouncer `session`) | Yes | **Supported** — the read-only `SET` lasts for the session |
| Transaction (e.g. PgBouncer `transaction`) | Higher | **Not supported** — see below |

`query_only=True` issues its read-only statement on the SQLAlchemy `connect` event, so
it applies to a physical connection. Under transaction-mode pooling a physical
connection can be handed to a different logical client between transactions, and that
session state is not guaranteed to survive the handoff. The failure is silent: a load
that believes it is read-only quietly stops being so.

With transaction-mode pooling, set `query_only=False` and enforce read-only access
with database grants instead, where it cannot be lost in a handoff.

### Measuring it

Set `diagnostics=True` on a partitioned load and the completion line reports how long
fetches spent queueing for a slot:

```
Partitioned SQL load completed strategy=offset partitions=4 rows=40 result_partitions=1
  elapsed=0.04s prepare_stmt=0.000s load_plan=0.031s
  gate_waits=0/4 gate_wait_total=0.000s gate_wait_max_seen=0.000s
```

`gate_waits=0/4` means none of the four fetches had to wait. A rising
`gate_wait_total` alongside a database that is not CPU-bound is the signature of the
failure above — concurrency queued rather than working. The other stage timers cannot
show this, because they are all measured *inside* the gate.

The fields are omitted entirely when no fetch ran in the logging process. A lazy load
returns a Dask graph before anything is fetched, and in a distributed run the fetches
happen in other processes; reporting `0.000s` there would say "no contention" when the
truth is "not measured here".

For lazy loads, read the counters after `.compute()` instead:

```python
from boti_data import fetch_gate_stats, reset_fetch_gate_stats

reset_fetch_gate_stats()
result = helper.dask.load().compute()

for gate_key, stats in fetch_gate_stats().items():
    print(gate_key, stats.waited_fetches, "/", stats.fetches, stats.total_wait_seconds)
```

`fetch_gate_stats()` reports the **calling process only**. On a distributed cluster each
worker accumulates its own counters, so run it on the workers (e.g. via
`client.run(fetch_gate_stats)`) to see the whole picture.

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
