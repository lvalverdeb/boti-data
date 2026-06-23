# Feature Specification: boti-data ETL Framework

**Feature Branch**: `main`

**Created**: 2026-06-23

**Status**: Active

**Input**: Requirements for a Python ETL framework sourcing data from relational databases (SQL) and landing it into columnar storage (Parquet), with incremental watermark tracking and Dask-first lazy evaluation.

---

## User Scenarios & Testing

### User Story 1 — Load SQL Data into DataFrames (Priority: P1)

As a data engineer, I can query a SQL database and receive results as a DataFrame, so that I can process and analyze the data without writing boilerplate connection management code.

**Why this priority**: Core capability — every downstream feature depends on data extraction.

**Independent Test**: Can be fully tested by pointing at a SQLite database file with known data, calling `DataHelper.load(filters={...})`, and verifying the returned DataFrame matches expected rows and columns.

**Acceptance Scenarios**:

1. **Given** a SQLite database with a `users` table containing 5 rows, **When** I call `DataHelper(config, table="users").load()`, **Then** I receive a `dd.DataFrame` with 5 rows.
2. **Given** a SQLite database with 10,000 rows in `events`, **When** I call `DataHelper(config, table="events").load()` with `return_type="auto"`, **Then** the gateway auto-detects the dataset as large and returns a `dd.DataFrame` (lazy).
3. **Given** a SQLite database with 5 rows in `events`, **When** I call `DataHelper(config, table="events").load()` with `return_type="auto"`, **Then** the gateway auto-detects the dataset as small and returns a `pd.DataFrame` (eager).
4. **Given** a SQL statement with a `field_map` that maps `db_name` to `semantic_name`, **When** I call `load()`, **Then** the result DataFrame's columns use semantic names.

---

### User Story 2 — Load Parquet Data into DataFrames (Priority: P1)

As a data engineer, I can read a partitioned Parquet dataset and receive a DataFrame, so that I can query columnar data without managing file discovery manually.

**Why this priority**: Core capability — parquet is the primary output format and a common input format.

**Independent Test**: Can be fully tested by writing a known DataFrame to a temp parquet directory, then reading it back with `ParquetReader` and verifying the data round-trips correctly.

**Acceptance Scenarios**:

1. **Given** a partitioned parquet dataset with 3 partition columns, **When** I call `ParquetDataResource.load_files()`, **Then** I receive a `dd.DataFrame` with all partitions unioned.
2. **Given** a parquet dataset with hive-style partitioning (`partition_date=YYYY-MM-DD/`), **When** I call with `parquet_start_date` and `parquet_end_date`, **Then** only files within the date range are loaded.
3. **Given** a parquet path that does not exist, **When** I call `ParquetDataResource.load_files()`, **Then** an empty DataFrame is returned (not an error).

---

### User Story 3 — Materialize SQL to Parquet (Priority: P1)

As a data engineer, I can materialize data from a SQL source into a partitioned Parquet dataset with a single orchestrated call, so that I can build ETL pipelines without gluing separate read/write steps together.

**Why this priority**: Primary ETL workflow — this is the main value proposition of the framework.

**Independent Test**: Can be fully tested by seeding a SQLite database, creating a `ParquetPipeline(source=DataHelper(...), destination=...)`, calling `materialize(reload=True)`, and verifying parquet files exist on disk and can be read back.

**Acceptance Scenarios**:

1. **Given** a `ParquetPipeline` with a `DataHelper` source and a temp directory destination, **When** I call `materialize(reload=True)`, **Then** parquet files are created at the destination path and a `ParquetMaterializationResult` is returned with the reloaded frame.
2. **Given** a `ParquetPipeline` with `partition_on=("partition_date",)`, **When** I call `materialize()`, **Then** the output is hive-partitioned by date.
3. **Given** an overwrite scenario, **When** I call `materialize()` twice with `overwrite=True`, **Then** the second run replaces all files and the result reflects the latest data.

---

### User Story 4 — Incremental Watermark-Based Loading (Priority: P2)

As a data engineer, I can incrementally load only new or changed records since the last run, so that repeated pipeline executions are fast and avoid processing historical data.

**Why this priority**: High value for production ETL where full reloads are impractical; depends on US1 and US3.

**Independent Test**: Can be fully tested by running a full load, recording the watermark, inserting new rows into the source, running the incremental load, and verifying only the new rows are fetched and written.

**Acceptance Scenarios**:

1. **Given** a source with 100 rows and no prior watermark, **When** I call `materialize()` on an incremental pipeline, **Then** all 100 rows are loaded and a watermark is persisted.
2. **Given** a watermark of `2026-06-01` and 0 new rows in the source, **When** I call `materialize()`, **Then** 0 rows are loaded, no parquet files are written, and the watermark is unchanged.
3. **Given** a watermark of `2026-06-01` and 5 new rows with later dates, **When** I call `materialize()`, **Then** only the 5 new rows are loaded and written, and the watermark advances to the max date among those 5 rows.
4. **Given** a `FileWatermarkStore` with a custom path, **When** I run an incremental pipeline, **Then** watermarks are persisted at the custom path and survive process restarts.

---

### User Story 5 — Filter with Pushdown Optimization (Priority: P2)

As a data engineer, I can apply filters to my data loads, and filters that the backend can handle natively are pushed down to the storage layer for efficiency.

**Why this priority**: Performance optimization; depends on US1.

**Independent Test**: Can be tested by applying a filter, inspecting that a SQL WHERE clause is generated (via diagnostics), and verifying the result is correctly filtered.

**Acceptance Scenarios**:

1. **Given** a SQL backend and a filter `{"status": "active"}`, **When** I call `load(filters={...})`, **Then** the SQL statement includes `WHERE status = :status` (pushdown).
2. **Given** a filter that combines pushdown and residual expressions, **When** I call `load()`, **Then** the pushdown part is applied at the backend and the residual part is applied in-memory, and the result is correct.
3. **Given** a parquet backend and a filter on a partition column, **When** I call `load()`, **Then** only the relevant partition directories are scanned (partition pruning).

---

### User Story 6 — Pipeline Enrichment (Priority: P3)

As a data engineer, I can attach optional enrichment steps between the load and write phases of a pipeline, so that I can transform or augment data before materialization.

**Why this priority**: Nice-to-have; depends on US3. Not required for basic ETL.

**Independent Test**: Can be tested by creating a custom `FrameEnricher` that adds a column, attaching it to a `SinkPipeline`, calling `write()`, and verifying the output contains the enriched column.

**Acceptance Scenarios**:

1. **Given** a `SinkPipeline` with a `FrameEnricher` that adds a `processed_at` timestamp column, **When** I call `write()`, **Then** the output dataset includes the `processed_at` column for every row.
2. **Given** a `SinkPipeline` without an enricher, **When** I call `write()`, **Then** the output is an exact copy of the source with no extra columns.

---

### Edge Cases

- What happens when the SQL query returns 0 rows? → Empty DataFrame, no files written.
- How does the system handle a missing watermark file on first run? → Treated as no prior watermark; full load is performed.
- What happens when parquet overwrite is True but the directory doesn't exist? → Directory is created.
- How are concurrent writes to the same FileWatermarkStore handled? → Thread-safe via `threading.Lock`.
- What happens when the data contains nulls in the watermark field? → Nulls are ignored by `max()`; watermark advances only on non-null values.

---

## Requirements

### Functional Requirements

- **FR-001**: System MUST load data from SQL databases (SQLite, MySQL, PostgreSQL) via SQLAlchemy.
- **FR-002**: System MUST load data from partitioned and non-partitioned Parquet datasets.
- **FR-003**: System MUST support Dask DataFrames as the default return type for all loads.
- **FR-004**: System MUST support pandas, PyArrow Table, and Polars DataFrames as explicit return types.
- **FR-005**: System MUST auto-detect dataset size when `return_type="auto"` and choose pandas (small) or dask (large).
- **FR-006**: System MUST support both structured mode (`statement` + `model`) and configured mode (`table=` at construction).
- **FR-007**: System MUST materialize DataFrames to partitioned Parquet datasets via `ParquetPipeline`.
- **FR-008**: System MUST support incremental loads with watermark persistence via `WatermarkStore`.
- **FR-009**: System MUST split filters into pushdown (backend) and residual (in-memory) components.
- **FR-010**: System MUST support field name translation from DB column names to semantic names via `field_map`.
- **FR-011**: System MUST support chunked IN-loading for large `field__in=[...]` lists (chunks of 900).
- **FR-012**: System MUST support semi-joins against existing DataFrames.
- **FR-013**: System MUST enforce read-only SQL connections by default (`query_only=True`).
- **FR-014**: System MUST protect against path traversal attacks in parquet file access.
- **FR-015**: System MUST support optional `FrameEnricher` transforms in pipeline execution.
- **FR-016**: System MUST support CSV and JSONL sinks in addition to Parquet.
- **FR-017**: System MUST provide both sync and async APIs for all major operations.
- **FR-018**: System MUST persist connection credentials as `pydantic.SecretStr` — never logged in plaintext.
- **FR-019**: System MUST support `worker_connection_env_var` for safe distributed DSN resolution on Dask workers.
- **FR-020**: System MUST provide diagnostics logging (plan, partition, client summaries) when `diagnostics=True`.

### Key Entities

- **DataHelper**: Public facade; entry point for all data loading. Wraps `DataGateway`. Provides engine-bound views (`.dask`, `.pandas`, `.polars`) that enforce compatible return types.
- **DataGateway**: Core orchestration engine. Resolves return type, execution mode, builds load requests, dispatches to backend-specific load implementations.
- **SqlDatabaseResource / AsyncSqlDatabaseResource**: SQL backend resource. Manages engine lifecycle, connection pooling, read-only enforcement.
- **ParquetDataResource**: Parquet backend resource. Handles file discovery via `pyarrow.dataset`, hive partition pruning, column projection.
- **DatacubeResource**: Datacube backend resource. Contract-based OLAP data access.
- **ParquetPipeline / SinkPipeline**: Write orchestration. Loads from a source (DataHelper or HybridDataset), enriches optionally, writes to a sink.
- **ParquetSink / CsvSink / JsonlSink**: Sink implementations that materialize DataFrames to disk.
- **WatermarkStore (FileWatermarkStore)**: Protocol for persisting watermark values. File-based implementation uses JSON with thread-safe writes.
- **FilterHandler**: Normalizes and splits filters into pushdown and residual components.
- **SqlPartitionedLoader / SqlPartitionPlanner / SqlPartitionExecutor**: Distributed SQL execution via partition splitting.
- **EngineRegistry**: Singleton cache for SQLAlchemy engines — prevents connection proliferation.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: A data engineer can load 1M rows from SQLite into a Dask DataFrame in under 10 seconds (local machine). *(Goal — no automated performance gate currently enforces this threshold.)*
- **SC-002**: A data engineer can materialize 100K rows from SQL to partitioned Parquet in a single `pipeline.materialize()` call. *(Goal — no automated performance gate currently enforces this threshold.)*
- **SC-003**: Incremental loads after a full load complete in under 2 seconds when no new data exists (0-row delta). *(Goal — functional behavior is verified, timing assertion not yet gated in CI.)*
- **SC-004**: Filter pushdown eliminates non-matching parquet partitions via partition pruning (`dataset.get_fragments(expression)`) and generates correct SQL WHERE clauses. *(Mechanism is implemented; "100% elimination" is a goal — exclusion coverage is verified by integration test, not quantified per-run.)*
- **SC-005**: The test suite (443 tests) passes with 0 failures across all backends.
- **SC-006**: All 29 security regression tests (parametrized) pass, confirming path traversal protection, credential safety, and engine isolation.

---

## Assumptions

- Users have Python 3.13+ and `uv` installed.
- Target databases support SQLAlchemy dialect drivers (`pymysql`, `asyncmy`, `sqlite3` built-in).
- Users have a local or remote Dask cluster available for distributed execution (optional).
- Parquet files use PyArrow as the engine (not fastparquet).
- Network filesystems (S3, GCS) accessed via `filesystem_profile` in `ConnectionCatalog`.
- Mobile and browser targets are out of scope — Python backend only.
- Maximum watermark field type is datetime (not integer-based watermarks, though API supports it).
- The watermark store JSON file is not shared across multiple concurrent pipeline instances (no distributed locking).
