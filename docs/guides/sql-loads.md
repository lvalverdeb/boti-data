# SQL loads

Everything that reads rows from a SQL database: the single-row repositories, the low-level resource, and `DataHelper`/`DataGateway` for dataframes. Driver installation and the backend-specific notes for ClickHouse, DuckDB and pgvector are in [Installation](../installation.md).

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

## Streaming a result

`iter_batches` never materialises the whole result. SQL sources read through a
server-side cursor, Parquet sources through dataset batches, and each batch is a
typed `pyarrow.RecordBatch` already renamed through the gateway's field map:

```python
with DataGateway(config, table="orders") as gateway:
    for batch in gateway.iter_batches(status="active", batch_size=50_000):
        sink.write(batch)          # about one batch of memory at any time

with SqlRepository(config, "orders") as repo:
    for batch in repo.iter_batches(columns=["id", "total"]):
        ...
```

`return_type="polars"` builds the frame from Arrow directly (no pandas on the
way), and `return_type="arrow"` needs no engine extra at all.

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

## Repository (lightest-weight, no dataframe dependencies)

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

### Multi-table atomic transactions

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

## Nearest-neighbour search (pgvector)

`boti_data.db.nearest_neighbors`/`vector_distance` build an `ORDER BY <distance> LIMIT k` statement against a pgvector column, with the query vector bound as a driver parameter — never string-interpolated into the SQL, regardless of how large the vector is. Requires `boti-data[pgvector]` (see [Installation](../installation.md)) and a table reflected via `SqlAlchemyModelBuilder`/`SqlRepository`/`DataGateway`.

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

## SQL resource (low-level)

```python
from boti_data import SqlDatabaseConfig, SqlDatabaseResource

config = SqlDatabaseConfig(connection_url="sqlite:///example.db", query_only=True)

with SqlDatabaseResource(config) as db:
    with db.session() as session:
        rows = session.execute(...)
```

## Gateway (mid-level)

```python
from boti_data import DataGateway, SqlDatabaseConfig

gateway = DataGateway(
    SqlDatabaseConfig(connection_url="sqlite:///example.db", query_only=True),
)
```

## DataHelper — local pandas

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

## DataHelper — local polars

```python
with DataHelper(config, table="sales") as helper:
    df = helper.polars.load(year=2024)
    summary = df.group_by("region").agg(pl.col("amount").sum())
```

## DataHelper — lazy Dask, no cluster

```python
with DataHelper(config, table="sales") as helper:
    ddf = helper.dask.load(year=2024)
    # Graph is not executed yet; chain transforms lazily
    result = ddf.groupby("region")["amount"].sum().compute()
```

## DataHelper — distributed Dask cluster

```python
from dask.distributed import LocalCluster
from boti_data import DataHelper, SqlDatabaseConfig

config = SqlDatabaseConfig(
    connection_url="mysql+pymysql://user:pass@host/mydb",
    query_only=True,
    worker_connection_env_var="DB_URL",
)

with DataHelper.session(cluster_factory=LocalCluster, cluster_kwargs={"n_workers": 4}) as client:
    with DataHelper(config, table="events") as helper:
        ddf = helper.dask.load(event_type="purchase", persist=True)
        result = ddf.groupby("user_id").size().compute()
```

## DataHelper — async service

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

## DataHelper — date-range load

```python
with DataHelper(config, table="transactions") as helper:
    # Inclusive date range; dt_field is the semantic field name
    df = helper.pandas.load_period("created_at", "2024-01-01", "2024-06-30")
```

## Raw `sql=` safety policy (eager SQL only)

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
