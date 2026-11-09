# Installation

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
pip install "boti-data[clickhouse]" # clickhouse-connect (sync only)
pip install "boti-data[duckdb]"     # duckdb-engine (sync only)
```

Parquet storage on `s3://` (including MinIO through `fs_type=s3`) needs `s3fs`, which `boti` 1.1.0 and later no longer install by default:

```bash
pip install "boti-data[s3]"         # boti[s3] -> s3fs
```

ClickHouse support is sync-only: there is no maintained async SQLAlchemy driver for ClickHouse, so `AsyncSqlDatabaseResource` rejects a ClickHouse connection URL — use `SqlDatabaseResource` instead. `SqlUnitOfWork`/`AsyncSqlUnitOfWork` also reject ClickHouse outright, since ClickHouse's transaction support is experimental, single-node, and MergeTree-only, with no real cross-table ACID guarantees to back a multi-table unit of work — use `SqlRepository`/`AsyncSqlRepository` directly, one table at a time, instead.

Verified against a real ClickHouse instance:

- `SqlRepository`/`AsyncSqlRepository` are a get/insert/update/delete-**by-primary-key** API, but MergeTree's `ORDER BY` is a sort/index prefix, not a uniqueness constraint — many real ClickHouse tables have several rows per `ORDER BY` key. Because of that, `get()`/`update()`/`delete()` each **refuse to run against a ClickHouse table unless called with `assume_unique_key=True`** (e.g. `repo.get(pk, assume_unique_key=True)`). This flag is a per-call caller assertion, not something boti-data verifies or enforces: it does not check the data, and if the key genuinely isn't unique, `get(key, assume_unique_key=True)` will still return one arbitrary matching row and `delete(key, assume_unique_key=True)`/`update(key, ..., assume_unique_key=True)` will still affect *every* row sharing that key, not just one. Only pass it once you've confirmed the key is actually unique for that table. `insert()` has no such parameter — appending a row is never ambiguous.
- `.update()` also fails outright on a table without `enable_block_number_column` enabled (`ALTER TABLE ... MODIFY SETTING enable_block_number_column = 1`, off by default) — ClickHouse raises `Lightweight updates are not supported` otherwise.
- `query_only=True` is enforced server-side (ClickHouse's `readonly` setting, applied via `set_client_setting` on connect), not just by boti-data's own read-only session guard.

DuckDB support is also sync-only (no maintained async driver, same as ClickHouse — `AsyncSqlDatabaseResource` rejects a DuckDB connection URL), but is otherwise a better fit than ClickHouse for `SqlRepository`'s by-primary-key model:

- **`SqlUnitOfWork`/`AsyncSqlUnitOfWork` are fully supported** — DuckDB has real multi-table ACID transactions (unlike ClickHouse), verified with an actual mid-transaction constraint violation correctly rolling back writes across two tables.
- **No `assume_unique_key`-style flag is needed** — DuckDB genuinely enforces `PRIMARY KEY`/`UNIQUE` constraints at the database level, and boti-data recovers the real key columns from DuckDB's catalog even though `duckdb-engine`'s own reflection can't see them (a known upstream limitation — it warns "doesn't yet support reflection on indices"). Without that recovery step, every DuckDB table would look keyless and `SqlRepository` would silently bind to an arbitrary first column instead of the real key — this was caught and fixed during implementation, not just assumed away.
- **`query_only=True` is enforced via a connect-time flag**, not a runtime statement: DuckDB's read-only mode (`duckdb.connect(read_only=True)`) can only be set when a connection is opened, so boti-data injects `connect_args={"read_only": True}` itself before the engine is even constructed, rather than configuring it afterward the way Postgres/MySQL/ClickHouse do.
- **DuckDB allows only one process to hold a database file open for read-write at a time** — opening a file in its default mode claims an exclusive lock on connect, even for a connection that only ever reads. `SqlPartitionedLoader` (which spawns one engine per worker process) requires `query_only=True` for a file-backed DuckDB DSN for exactly this reason — otherwise the first worker to connect would block every other worker, regardless of whether any of them write. Concurrent read-only access from multiple processes works fine. In-memory (`duckdb:///:memory:`) DSNs are rejected for distributed loading the same way SQLite's are, since each worker process would get its own independent, empty database.

If a Postgres table has a [pgvector](https://github.com/pgvector/pgvector) `vector` column, install `boti-data[pgvector]` too — this registers pgvector's `Vector` type with SQLAlchemy's reflection so `SqlAlchemyModelBuilder`/`SqlRepository`/`DataGateway` map that column to a usable `list[float]` (with its dimension preserved) instead of an unreflectable `NullType`:

```bash
pip install "boti-data[pgvector]"
```

The base install is deliberately light: single-row transactional access (`SqlRepository`/`AsyncSqlRepository`, see the [SQL loads guide](guides/sql-loads.md)) works out of the box, with no dask/pandas/polars pulled in. The dataframe-shaped, bulk-oriented side of the package (`DataGateway`/`DataHelper`, `ParquetPipeline`, `HybridDataset`, the sink pipelines, ...) needs the `dataframes` extra:

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
