# Security

boti-data inherits its security posture from `boti` and adds SQL-specific guards on top. The
regression suite under `tests/security/` (30 tests, marked `security_regression`) pins every
fix below; CI runs it on every push.

## Sandboxed paths and filesystems

Every file-backed resource (`ParquetDataResource`, the sinks, the watermark stores) is a
`boti` `SecureResource`: paths are resolved against the project root and refused when they
escape it, NUL bytes are rejected for every scheme, and remote filesystems built from a
`ConnectionCatalog` profile go through boti's SSRF-guarded `fsspec` adapters. Injected
filesystems are honoured on remote paths rather than silently replaced.

## Read-only database access

`query_only=True` (the default for `SqlRepository`/`AsyncSqlRepository`) makes the engine
read-only at the connection level, enforced server-side where the backend supports it:
PostgreSQL and MySQL set the session read-only on connect, ClickHouse applies its `readonly`
setting, and DuckDB opens the file with `read_only=True`. It always overrides a passed-in
config's own `query_only`, so a repository never inherits write access nobody asked for.
Transaction-mode poolers can lose that session state between hand-offs; see
[Sizing connections](distributed.md#sizing-connections-and-fetch-concurrency).

## Raw SQL

`DataGateway` accepts raw `sql="..."` only on eager paths, validated as a single read-only
`SELECT`/`WITH` statement, and only when the call site opts in with `allow_raw_sql=True`
under the default `raw_sql_policy="readonly_opt_in"`. `raw_sql_policy="disabled"` blocks it
outright. Lazy and partitioned loads always take `statement` + `model`, so every query a worker
runs is compiled by SQLAlchemy with bound parameters. See
[the SQL loads guide](sql-loads.md#raw-sql-safety-policy-eager-sql-only).

## Credentials in distributed runs

Workers never receive a `SqlDatabaseConfig`. They get a minimal `WorkerSqlConfig`, and when
`worker_connection_env_var` is set they read the DSN from their own environment, so no
credential crosses the scheduler or lands in a task payload or a worker log. The cluster-wide
fetch gate is named by a hash of the engine identity for the same reason: the name reaches the
scheduler and its dashboard. See [the distributed guide](distributed.md#the-pickleable-setting-in-distributed-systems).

## Pickling is opt-in twice

A resource pickles only when its config sets `allow_pickle=True`, and unpickles only in a
process that has opted in with `BOTI_ALLOW_TRUSTED_RESOURCE_UNPICKLE=1` or
`trusted_unpickle_scope()`. The gate is an acknowledgement, not a sanitiser: `pickle` runs a
payload's `__reduce__` before any class-level hook can inspect it, so unpickling a payload from
an untrusted source is still arbitrary code execution. Enable it only where every producer is
trusted.

## Logging

The `boti` logger redacts secrets, and diagnostics lines log counts, engine names and timings,
never row data.

Further reading: [security hardening notes](../proposals/SECURITY_HARDENING.md).
