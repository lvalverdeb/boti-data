# boti-data

`boti-data` is the **data access and transformation layer** of the Boti ecosystem: a secure,
SQLAlchemy-based library that gives teams one reusable interface for structured data across
databases, Parquet datasets, schema-controlled transformations, and partitioned or distributed
loading.

It builds on [`boti`](https://github.com/lvalverdeb/boti) (security, filesystems, lifecycle)
and [`boti-dask`](https://github.com/lvalverdeb/boti-dask) (the Dask runtime), and owns SQL,
Arrow and dataframe shaping. Full documentation: **https://lvalverdeb.github.io/boti-data/**

- Single-row access: `SqlRepository`, `AsyncSqlRepository`, `SqlUnitOfWork`, reflected models.
- Dataframe gateway: `DataGateway`/`DataHelper` returning pandas, polars, Arrow or Dask, with
  sticky filters, dry-run and preview.
- Backends: MySQL, PostgreSQL (with pgvector nearest-neighbour search), ClickHouse and DuckDB
  (sync only), SQLite for tests.
- Partitioned, distributed SQL loading with a cluster-wide fetch gate and pickle-safe resources.
- Parquet resources and pipelines, CSV/JSONL/Parquet sinks, hybrid historical-plus-live datasets.

## Installation

```bash
pip install boti-data                 # single-row API only; no pandas or dask
pip install "boti-data[pandas]"       # DataGateway/DataHelper/pipelines returning pandas; no Dask on the path
pip install "boti-data[polars]"       # polars results built from Arrow (pandas comes along for shaping)
pip install "boti-data[dask]"         # lazy and distributed loads: dask + distributed + boti-dask
pip install "boti-data[dataframes]"   # all three engines
```

The engines are separate extras, so a pandas-only service never installs or imports Dask;
`boti_data` detects an engine from the object it is handed and loads it on first use.

Install the driver matching your connection URL's scheme, and `s3` for `s3://` Parquet storage:

```bash
pip install "boti-data[mysql]"        # asyncmy + pymysql
pip install "boti-data[postgres]"     # asyncpg + psycopg[binary]
pip install "boti-data[pgvector]"     # registers pgvector's Vector type for reflection
pip install "boti-data[clickhouse]"   # clickhouse-connect (sync only)
pip install "boti-data[duckdb]"       # duckdb-engine (sync only)
pip install "boti-data[s3]"           # boti[s3] -> s3fs
pip install "boti-data[adbc]"         # ADBC drivers: Arrow straight from PostgreSQL/SQLite on read-only fetches
```

Loads stream too: `DataGateway.iter_batches(...)`, `SqlRepository.iter_batches()` and
`AsyncSqlRepository.iter_batches()` yield Arrow record batches through a server-side cursor,
so a scan of any size costs about one batch of memory. Partition fetches build Arrow in one
columnar pass (3.4x faster than 1.3.x on a 1M-row SQLite fetch), `return_type="polars"` is
built from that Arrow with no pandas detour, and `ParquetDataResource.scan_polars()` returns
a lazy polars frame over the resource's filesystem.

Every name importable from `boti_data` loads lazily, so `import boti_data` costs a few
milliseconds and never touches dask, pandas or polars until a dataframe-shaped name is used.
Backend-specific notes (ClickHouse's `assume_unique_key`, DuckDB's read-only file locking) are
in the [installation guide](https://lvalverdeb.github.io/boti-data/installation/).

## Examples

### Repository: single-row CRUD, no dataframe dependencies

`query_only` defaults to `True` and always overrides the config's own value, so writes raise
until you opt in explicitly.

```python
from boti_data import SqlDatabaseConfig, SqlRepository, SqlUnitOfWork

config = SqlDatabaseConfig(connection_url="sqlite:///example.db")

with SqlRepository(config, "users", query_only=False) as users:
    users.insert({"id": 1, "name": "Ada"})
    row = users.get(1)                       # -> dict | None
    users.update(1, {"name": "Grace"})
    deleted = users.delete(1)                # -> bool

with SqlUnitOfWork(config, query_only=False) as uow:
    uow.repository("audit_log").insert({"event": "enrol"})
    uow.repository("embeddings").insert({"subject_id": 1, "vector": [0.1, 0.2]})
# both rows commit on clean exit, or neither does on an exception
```

`AsyncSqlRepository` and `AsyncSqlUnitOfWork` mirror these as async context managers.

### DataHelper: one source, any output engine

```python
from boti_data import DataHelper, SqlDatabaseConfig

config = SqlDatabaseConfig(
    connection_url="postgresql+psycopg://user:pass@host/mydb",
    query_only=True,
)

with DataHelper(config, table="orders") as helper:
    df = helper.pandas.load(status="confirmed")                   # pandas.DataFrame
    pl_df = helper.polars.load(status="confirmed")                # polars.DataFrame
    ddf = helper.dask.load(status="confirmed")                    # lazy dask.dataframe.DataFrame
    period = helper.pandas.load_period("created_at", "2024-01-01", "2024-03-31")
    table = helper.load(status="confirmed", return_type="arrow")  # pyarrow.Table
```

In an async service, `await helper.pandas.aload(...)`. On a cluster, open the session first:

```python
with DataHelper.session(scheduler_address="tcp://scheduler:8786") as client:
    with DataHelper(config, table="events") as helper:
        result = helper.dask.load(region="EU", persist=True).groupby("customer_id").size().compute()
```

### ParquetPipeline: materialise a load into a Parquet dataset

```python
from boti_data import DataHelper, ParquetPipeline

helper = DataHelper(backend="sqlalchemy", connection_url="sqlite:///events.db", table="source_events")

pipeline = ParquetPipeline(
    helper,
    {"backend": "parquet", "storage_path": "/data/source_events", "partition_on": ["partition_date"]},
    date_field="event_date",
)
result = pipeline.materialize(
    filters={"status__exact": "active"},
    reload=True,
    reload_options={"filters": {"partition_date__exact": "2026-04-17"}},
)
frame = result.frame
```

`SinkPipeline` writes the same loads through `CsvSink`, `JsonlSink` or any registered sink.

## Documentation

| Topic | Where |
| --- | --- |
| Installation, drivers and extras | [installation](https://lvalverdeb.github.io/boti-data/installation/) |
| SQL loads: repositories, gateway, engines, raw SQL policy | [guides/sql-loads](https://lvalverdeb.github.io/boti-data/guides/sql-loads/) |
| Parquet sources, pipelines, hybrid datasets, catalogues | [guides/parquet](https://lvalverdeb.github.io/boti-data/guides/parquet/) |
| Sinks and enrichment | [guides/sinks](https://lvalverdeb.github.io/boti-data/guides/sinks/) |
| Distributed loading, credentials on workers, sizing connections | [guides/distributed](https://lvalverdeb.github.io/boti-data/guides/distributed/) |
| Datacube backend | [guides/datacube](https://lvalverdeb.github.io/boti-data/guides/datacube/) |
| Security posture | [guides/security](https://lvalverdeb.github.io/boti-data/guides/security/) |
| API reference: `db` | [api/db](https://lvalverdeb.github.io/boti-data/api/db/) |
| API reference: gateway and helper | [api/gateway](https://lvalverdeb.github.io/boti-data/api/gateway/) |
| API reference: parquet | [api/parquet](https://lvalverdeb.github.io/boti-data/api/parquet/) |
| API reference: pipelines and sinks | [api/pipelines](https://lvalverdeb.github.io/boti-data/api/pipelines/) |
| API reference: filters | [api/filters](https://lvalverdeb.github.io/boti-data/api/filters/) |
| API reference: watermarks, enrichment, datacube (experimental) | [api/watermark](https://lvalverdeb.github.io/boti-data/api/watermark/), [api/enrichment](https://lvalverdeb.github.io/boti-data/api/enrichment/), [api/datacube](https://lvalverdeb.github.io/boti-data/api/datacube/) |
| Benchmarks and baseline | [benchmarks](https://lvalverdeb.github.io/boti-data/benchmarks/) |
| Changelog and versioning policy | [CHANGELOG.md](CHANGELOG.md) |

Build the site locally with `make docs` (strict) or `make docs-serve`.

## Public API and versioning

Every name exported from `boti_data` is listed in `boti_data.STABLE` or `boti_data.EXPERIMENTAL`.
Stable names change shape only in a major release and are deprecated for at least one minor
release first, with a `DeprecationWarning` naming the removal version and the replacement.
Experimental names (enrichment, datacube, hybrid datasets, watermarks, vector search, the
partitioned loader internals) may change in any minor release. See the
[versioning policy](CHANGELOG.md#versioning-policy).

## Security

Every file-backed resource is a `boti` `SecureResource`: paths are sandboxed to the project
root, NUL bytes are rejected on every scheme, and remote filesystems go through SSRF-guarded
adapters. Database access is read-only by default and enforced server-side where the backend
allows it. Raw SQL is eager-only, validated as a single read-only statement, and opt-in per
call. Workers never receive credentials: with `worker_connection_env_var` they read the DSN
from their own environment, and the cluster-wide fetch gate is named by a hash, not the DSN.
Unpickling a resource needs an explicit trust opt-in in the receiving process. Thirty security
regression tests pin these behaviours in CI. Details:
[security guide](https://lvalverdeb.github.io/boti-data/guides/security/).

## Compatibility

Every push, and weekly, the compatibility-matrix workflow runs the suite against the floor,
locked and latest releases of `boti` and `boti-dask`. The floors are the oldest versions the
suite passes on; locked is what `uv.lock` resolves; latest is whatever PyPI serves that day.

| Dependency | Floor | Locked | Latest |
| --- | --- | --- | --- |
| `boti` | 1.2.0 | 1.2.0 | weekly |
| `boti-dask` | 1.0.0 | see `uv.lock` | weekly |

With `boti-dask` below 1.1.0 the partition-fetch gate bounds each worker process rather than
the cluster (see the [distributed guide](https://lvalverdeb.github.io/boti-data/guides/distributed/#sizing-connections-and-fetch-concurrency)).

## Relationship to `boti`

`boti-data` depends on `boti` for logging, resource lifecycle, secure I/O helpers and
project/environment utilities. If you only need the runtime primitives, install `boti`. If you
need the data access and transformation layer, install `boti-data` or `boti[data]`.

## Development

```bash
make install-dev      # uv sync --group dev
make verify           # the checks CI runs: lint, tests, security tests, spec counts
make bench            # benchmark suite (see benchmarks/README.md)
make docs             # build the documentation site, strict
```

Publishing instructions are in the [deployment notes](https://lvalverdeb.github.io/boti-data/proposals/DEPLOYMENT/).
