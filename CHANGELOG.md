# Changelog

All notable changes to boti-data are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning policy

- Every name exported from `boti_data` is listed in exactly one of
  `boti_data.STABLE` and `boti_data.EXPERIMENTAL`.
- **Stable** names keep their signature and behaviour across minor releases.
  A stable name is removed or changed incompatibly only in a major release,
  and is first deprecated for at least one minor release: accessing it emits
  a `DeprecationWarning` naming the removal version and the replacement.
- **Experimental** names may change in any minor release. They are marked as
  such in the reference documentation and in this file when they change.
- Patch releases contain fixes only. Minor releases add features and
  deprecations. Major releases (2.0 onwards) remove what was deprecated and
  may change stable behaviour, with a migration guide.
- Floors on `boti` and `boti-dask` are raised in minor releases and recorded
  under "Changed", together with the compatibility matrix in the README.

## [Unreleased]

## [1.4.0] - 2026-11-09

Targeting 1.4.0.

### Added

- ADBC partition fetches behind the read-only guards (`boti-data[adbc]`):
  with the driver installed and `query_only=True`, PostgreSQL and SQLite
  partitions are fetched through ADBC, which returns Arrow straight from the
  wire protocol. The ADBC session gets the same read-only statement the
  SQLAlchemy engine issues, only the planner's compiled partitions are sent
  (psycopg placeholders are rewritten to `$n`), one connection is opened per
  fetch and closed with it (the workers' `NullPool` rule, so the fetch gate
  bounds open connections), and a driver error falls back to SQLAlchemy for
  that partition. `adbc=False` opts out.
- Arrow-native partition fetch (`boti_data.db.arrow_fetch`): cursor results
  become `pyarrow.Table` in one columnar pass, using the driver's own Arrow
  reader where there is one (DuckDB) and raw DBAPI rows elsewhere, with a
  vectorised `pyarrow.compute` cast before any per-value coercion. A 1M-row
  SQLite partition fetch went from 2.93 s to 0.87 s; the Arrow and pandas
  paths produce identical frames.
- Streaming results: `DataGateway.iter_batches(...)`,
  `SqlRepository.iter_batches()` and `AsyncSqlRepository.iter_batches()` yield
  Arrow record batches through a server-side cursor, holding about one batch
  of memory however large the result.
- Engine extras `pandas`, `polars` and `dask`; `dataframes` is now their
  union. No module imports dask, polars or boti-dask at import time
  (`boti_data._engines` loads them on first use), so a pandas-only install
  never loads Dask, and a missing engine is reported as the extra to add.
- polars-native path: `return_type="polars"` is built from Arrow with no
  pandas detour for SQL and Parquet loads; `ParquetDataResource.scan_polars()`
  returns a lazy polars frame over the resource's filesystem.
- Metrics and tracing (R8): `boti_data.telemetry` with a no-op default, an
  in-memory provider for tests and an OpenTelemetry provider behind the new
  `otel` extra, selected with `telemetry.configure()` or
  `BOTI_DATA_TELEMETRY=otel`. Spans around every gateway load and partition
  fetch, a histogram of fetch-gate waits, and pool checkout counter and
  checked-out gauge from SQLAlchemy pool events on every engine, keyed on the
  connection-URL digest, never the DSN. See the "Metrics and tracing" guide.
- Cluster-wide partition fetch gate: on a Dask cluster the gate is
  `boti_dask.cluster_semaphore`, a scheduler-side semaphore shared by every
  worker, so `max_concurrent_fetches` bounds the whole cluster rather than
  each worker process. The gate is keyed on a hash of the database identity
  alone, never on the limit, so two loads with different limits share one gate
  (first limit wins, warning logged). Without a running client, or with
  boti-dask below 1.1.0, the process-wide gate applies as before.
- Compatibility-matrix workflow testing the floor, locked and latest releases
  of `boti` and `boti-dask` on every push, weekly, and on an
  `upstream-release` repository dispatch; matrix published in the README.
- `s3` extra (`boti-data[s3]`) delegating to `boti[s3]` for `s3fs`, which
  boti 1.1.0 and later no longer install by default.
- Public API contract: `boti_data.STABLE` and `boti_data.EXPERIMENTAL`
  stability tuples covering every exported name, `boti_data.DEPRECATED` with
  the `boti_data._contract.deprecated` helper wired into the lazy module
  `__getattr__`, and this changelog.
- Benchmark suite under `benchmarks/` (eight pytest-benchmark cases runnable
  against SQLite and local Parquet), a committed baseline, `make bench` and
  `make bench-compare`, and a workflow that fails a release when a median
  regresses by more than 20 percent.
- Reference documentation site built with mkdocs-material and mkdocstrings,
  versioned with mike on release tags; README cut to installation, three
  examples and links.
- Incremental loads: `DataGateway.load(since=...)` / `aload(since=...)`
  filter by a stored watermark and advance it after a successful load;
  `SqlWatermarkStore` keeps the watermark in a `boti_watermarks` table next to
  the data. Partitioned SQL loads are resumable: `SqlPartitionedLoader
  .load_request(request, checkpoint=PlanCheckpoint(store, key))` persists
  each finished partition, and `resume=True` on the request fetches only the
  partitions still pending (`SqlPartitionPlan.to_json()`/`from_json()`,
  `FileCheckpointStore`). `UpsertSink` merges a frame into a SQL table with
  `INSERT ... ON CONFLICT` (PostgreSQL, SQLite), `INSERT OR REPLACE` (DuckDB)
  or `ON DUPLICATE KEY UPDATE` (MySQL), registered as the `"upsert"` sink.
  Guide: `docs/guides/incremental-loads.md`.

### Security

Findings of the 2026-11 audit, each with a regression test under
`tests/security/`:

- Partition values taken from the data (`partition_on` columns) are validated
  before any sink writes: a value that is empty, `.`, `..`, or contains `/`,
  `\` or NUL fails the write. Such a value previously became a directory
  name and could place files outside the project root.
- A `parquet_storage_path` with a scheme other than `file://` is refused when
  the resource's filesystem is local. The scheme was stripped and the rest
  read as a local path, bypassing the sandbox on the Dask load path.
- **`overwrite=False` now appends.** Dask names output files `part.N` from
  zero on every write, so a second write into an existing dataset replaced
  the first's files. Appended files carry a unique token in their name.
- The raw-SQL guard also rejects `INTO`, `SET`, `set_config`, `RESET`,
  `LOAD`, `INSTALL`, `EXPORT`, `IMPORT`, `lo_import`, `lo_export`, `setval`,
  `nextval` and backend-termination functions. The guard remains a heuristic;
  database-enforced read-only (`query_only=True`) is the control to rely on.
- **DuckDB under `query_only=True` connects with `enable_external_access`
  off**, locked at connect time. `read_only` protects the database file only;
  table functions such as `read_text()` and `read_csv()` could still read
  local files and, through httpfs, the network. Reading external files from
  a query-only DuckDB connection now needs `query_only=False`.
- Async worker engines (`AsyncSqlPartitionedLoader`) now receive the same
  read-only session statement as sync worker engines; they validated support
  but never applied it.

Medium findings of the same audit, tests under
`tests/security/test_regressions_audit_mediums.py`:

- `CsvSink`/`JsonlSink` with a remote `storage_path` and no injected `fs`
  build the filesystem through boti's `FilesystemConfig`, so its endpoint
  validation (private and reserved addresses) applies. The previous
  `fsspec` fallback accepted any URL, including a loopback HTTP server.
- `ParquetSink` writes to the path the resource resolved against the project
  root. A relative `parquet_storage_path` was written (and with
  `overwrite=True`, removed) relative to the working directory instead.
- Engine registry keys and worker engine identities include a digest of
  `connect_args` and `execution_options`. Two configs with one DSN but
  different TLS settings shared an engine, the second silently getting the
  first's connections.
- The read-only `connect` listener is attached before a new engine is
  registered (`EngineRegistry.get_or_create(..., configure=)`); it was
  attached afterwards, leaving a window in which another holder could check
  out an unconfigured pooled connection.
- Dask `contains`, `not_contains` and `icontains` filters match literally,
  as the SQL twin's escaped `LIKE` does; they treated the value as a regex.
  `regex`/`iregex` patterns pass `validate_regex_pattern` on both backends.
- The Parquet append guard unifies every existing file's schema, not the
  first file's, so an append can no longer narrow a column a later
  partition introduced.
- `SqlPartitionSpec.key()` hashes the JSON form of the parameters, so a
  checkpoint reloaded from disk (dates stored as ISO strings) maps to the
  same partition keys and resumes instead of refetching everything.

Low findings of the same audit:

- Overwrites stage into a per-run `.staging-<token>` directory, so two
  overwriters of one target no longer delete each other's in-progress output.
  Leftover staging directories from crashed runs are reported, not removed.
  **Overwriting a bucket root is refused**, since its staging sibling would
  be another bucket.
- The unified read schema of an evolved Parquet dataset is cached per file
  list, so every footer is read once per dataset, not once per load.
- Checkpoints store `date`, `datetime`, `time` and `Decimal` parameters
  with their type and restore them as such; a resumed partition bound ISO
  strings where the plan had bound dates.
- `columns=` is resolved through the mapper like filter fields are; a name
  such as `__table__` widened the projection to every column.
- `in_chunk_size` must be at least 1 and `in_chunk_concurrency` between 1
  and 64; any integer was accepted and became the thread count.
- `SqlRepository.update()` rejects keys that are not mapped columns and keys
  of primary-key columns; any attribute name was assigned.
- The security guide documents what a pickled resource carries (remote
  filesystem credentials) and that checkpoint plans hold bound filter values.

### Changed

- `boti-dask` floor raised to 1.3.0: the fetch gate relies on
  `cluster_semaphore(strict=False)`, and the shared-session rebind and
  primitives fixes of that release.
- `boti` floor raised to 1.2.0: `ParquetDataResource` resolves a relative
  `parquet_storage_path` against the project root, which boti anchors there
  only from 1.2.0 (1.0.x anchored to the working directory).
- README "Sizing connections and fetch concurrency" rewritten for the
  cluster-wide gate.

### Fixed

- Async Arrow and polars SQL loads (`aload(return_type="arrow" | "polars")`)
  raised a `TypeError` after the Arrow-fetch change: the async loader still
  called the old row-list finaliser. Both paths now share one finaliser, and
  the async path is tested.
- Reading a Parquet dataset whose files disagree on a column's type in a way
  Arrow cannot promote (int64 in some files, strings in others) failed, since
  1.3.x on the first non-parsing value and since the schema-evolution read on
  unification. Such columns are now read as string, eagerly and through Dask,
  with a warning naming them; every value is kept.
- Boolean coercion on pandas 3.1: a column whose values all miss the literal
  map (floats such as `1.0`) came back as a null-typed Arrow series and the
  numeric fallback could not assign into it. The fallback now works on an
  object array. Caught by the compatibility matrix's "latest" jobs.

- `DataGateway` is pickleable again on boti 1.0.11 and later: the boti
  `Logger` now holds a `threading.Event`, so the gateway's post-processor
  drops its logger on pickle and `DataGateway.__setstate__` re-attaches the
  restored resource's logger.
- Parquet discovery tests no longer patch the removed
  `boti.core.secure_io.is_secure_path`.

## [1.3.11] - 2026-09-22

Last release before this changelog was kept. Sinks honour an injected
filesystem on remote paths and reject NUL bytes for every scheme; the
`dataframes` extra is named in the error when its dependencies are missing.

[Unreleased]: https://github.com/lvalverdeb/boti-data/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/lvalverdeb/boti-data/compare/v1.3.11...v1.4.0
[1.3.11]: https://github.com/lvalverdeb/boti-data/releases/tag/v1.3.11
