# Tasks: boti-data ETL Framework

**Input**: Design documents from `specs/boti-data-etl/`

**Prerequisites**: `spec.md`, `plan.md`, `constitution.md`

**Organization**: Tasks are grouped by user story. The entire framework is already implemented (v1.0.1). This task list represents the build order from scratch, useful for incremental feature planning or audit of what exists.

---

## Phase 1: Foundation (Shared Infrastructure)

**Purpose**: Project scaffolding, core abstractions, and base protocols

- [x] T001 Initialize Python project with `pyproject.toml`, dependency management (uv)
- [x] T002 Configure ruff linting, mypy types, pytest with asyncio mode
- [x] T003 Define core types: `BackendName`, `BackendConfig`, `BackendResource`, `FrameResult`, `ExecutionMode`, `ReturnType` in type stubs
- [x] T004 Create `SqlDatabaseConfig` with `pydantic.SecretStr`, pool settings, factory methods (`from_env`, `from_settings`)
- [x] T005 [P] Create `ParquetDataConfig` with storage path, partition, filesystem settings
- [x] T006 [P] Create `DatacubeConfig` with contract model
- [x] T007 Create `EngineRegistry` — singleton cache for SQLAlchemy engines
- [x] T008 Create `SecureResource` base — path traversal protection, null-byte detection, `allowed_paths` validation

---

## Phase 2: Filters & Expressions (Blocking Prerequisite)

**Purpose**: Filter expression tree that all backends use

- [x] T009 Create filter expression models: `Expr`, `And`, `Or`, `Not`, `ColOp` in `filters/expressions.py`
- [x] T010 [P] Create `FilterHandler` with `split_pushdown_and_residual()` in `filters/handler.py`
- [x] T011 [P] Create filter utilities: `normalize_filters` in `filters/utils.py`
- [x] T012 Create Arrow-native filter kernels in `filters/arrow_kernels.py`

**Checkpoint**: Foundation ready — filter expressions and pushdown splitting work.

---

## Phase 3: User Story 1 — Load SQL Data into DataFrames (Priority: P1) ✅

**Goal**: Load data from SQL databases via structured and configured modes

### Implementation

- [x] T013 [P] Create `SqlDatabaseResource` — engine/session management, read-only enforcement
- [x] T014 [P] Create `AsyncSqlDatabaseResource` — async engine/session
- [x] T015 Create `SqlAlchemyModelBuilder` — reflect tables, build declarative models
- [x] T016 Create `SqlModelRegistry` — cache reflected models
- [x] T017 Create `SqlPartitionPlanner` — plan partition splits (id range, date range)
- [x] T018 Create `SqlPartitionedLoader` — orchestrate partitioned loads
- [x] T019 Create `SqlPartitionExecutor` — execute partitions with `ThreadPoolExecutor`
- [x] T020 Create `DataGateway.core` — main orchestration: `_resolve_execution_plan`, `_execute_sync`, `load()`, `aload()`
- [x] T021 Create `gateway/requests.py` — `SqlLoadRequest`, `ParquetLoadRequest` models
- [x] T022 Create `gateway/loaders.py` — `load_sql()`, `load_sql_partitioned()` implementations
- [x] T023 Create `gateway/normalization.py` — `LOAD_CONTROL_KEYS`, filter normalization
- [x] T024 [P] Create `DataHelper` facade in `helper.py` — `load()`, `aload()`, `.dask`, `.pandas`, `.polars` views
- [x] T025 [P] Create `gateway/frame_strategies.py` — DaskFrameStrategy, PandasFrameStrategy, etc.
- [x] T026 Create `DataGateway.from_config()`, `DataGateway.from_backend()` constructors
- [x] T027 Create `gateway/sql_guard.py` — raw SQL read-only validation
- [x] T028 Create `field_map.py` — DB → semantic name translation
- [x] T029 Add `semi_join` support to DataGateway
- [x] T030 Add `load_period` support to DataGateway

### Tests

- [x] T031 Test SQL load with SQLite — structured mode, configured mode, filters
- [x] T032 Test return type auto-detection — small (pandas) vs large (dask)
- [x] T033 Test field_map translation
- [x] T034 Test semi-join against known DataFrames

**Checkpoint**: User Story 1 complete — SQL data loads work with both modes.

---

## Phase 4: User Story 2 — Load Parquet Data (Priority: P1) ✅

**Goal**: Read partitioned and non-partitioned parquet datasets

### Implementation

- [x] T035 Create `ParquetDataResource` in `parquet/resource.py` — `load_files()`, `load_arrow()`, `load_filtered()`
- [x] T036 Implement `_resolve_files_to_load()` — file discovery via `pyarrow.dataset` with hive partition pruning
- [x] T037 Implement `_discover_all_files()` — full dataset scan
- [x] T038 Implement `_discover_partitioned_files()` — date-range filtered discovery
- [x] T039 Create `ParquetReader(DataHelper)` in `parquet_reader.py` — parquet-specific facade
- [x] T040 Create `load_parquet()` in `gateway/loaders.py` — parquet backend implementation
- [x] T041 Implement `_empty_ddf()` — schema-preserving empty DataFrame for missing paths

### Tests

- [x] T042 Test parquet load — partitioned dataset, column projection, filter pushdown
- [x] T043 Test empty dataset — path doesn't exist → empty DataFrame
- [x] T044 Test date range filtering — only matching partitions loaded

**Checkpoint**: User Story 2 complete — parquet data loads work.

---

## Phase 5: User Story 3 — Materialize SQL to Parquet (Priority: P1) ✅

**Goal**: Full ETL pipeline from SQL source to partitioned parquet destination

### Implementation

- [x] T045 Create `SinkPipeline` in `pipelines/base.py` — `load()`, `write()`, `awrite()`
- [x] T046 Add materialization load option enforcement: force `return_type="dask"`, `execution_mode="lazy"`
- [x] T047 Create `ParquetPipeline(SinkPipeline)` — `to_parquet()`, `from_parquet()`, `materialize()`
- [x] T048 Create `ParquetSink` in `pipelines/sinks.py` — `to_dask_frame()`, `prepare_partitioned_frame()`, `ddf.to_parquet()`
- [x] T049 Implement overwrite logic in `ParquetSink.write()` — `fs.rm(target_path)` before write
- [x] T050 Create `CsvSink` in `pipelines/sinks.py:127`
- [x] T051 Create `JsonlSink` in `pipelines/sinks.py:292`
- [x] T052 Create `SinkRegistry` in `pipelines/registry.py`
- [x] T053 Implement `_materialize_full()` — non-incremental materialization flow
- [x] T054 Add `close()` and context manager support to sinks
- [x] T055 Create `SinkWriteResult` and `ParquetMaterializationResult` models

### Tests

- [x] T056 Test `ParquetPipeline.materialize()` with SQLite source — verify parquet files on disk
- [x] T057 Test materialize with `reload=True` — verify re-read matches source
- [x] T058 Test overwrite — second run replaces files, correct result
- [x] T059 Test CSV and JSONL sinks
- [x] T060 Test `SinkRegistry` factory methods
- [x] T061 Test async `amaterialize()` path

**Checkpoint**: User Story 3 complete — full ETL materialization works.

---

## Phase 6: User Story 4 — Incremental Watermark Loading (Priority: P2) ✅

**Goal**: Incremental pipeline execution with watermark persistence

### Implementation

- [x] T062 Define `WatermarkStore` protocol in `watermark/store.py`
- [x] T063 Create `FileWatermarkStore` — thread-safe JSON file persistence
- [x] T064 Create `build_incremental_filters()` in `watermark/incremental.py`
- [x] T065 Create `advance_watermark()` — compute max of watermark field across all frame types
- [x] T066 Create `IncrementalResult` model
- [x] T067 Implement `_materialize_incremental()` — read watermark → build filter → write → reload → advance
- [x] T068 Implement `DataHelper.load_incremental()` — facade-level incremental load
- [x] T069 Add `ParquetPipeline.incremental()` — class factory for incremental pipelines
- [x] T070 Handle edge case: no watermark on first run → full load
- [x] T071 Handle edge case: 0 rows returned → skip reload, don't advance watermark
- [x] T072 Guard `from_parquet()` reload behind `if result.files` check (fixes "No files found" warning)

### Tests

- [x] T073 Test `FileWatermarkStore` — read/write/clear, thread safety
- [x] T074 Test `advance_watermark` with pd.DataFrame, dd.DataFrame, pa.Table, pl.DataFrame
- [x] T075 Test `build_incremental_filters` — gt, gte operators
- [x] T076 Test `DataHelper.load_incremental()` — full → incremental → no-new-data → new-data
- [x] T077 Test `ParquetPipeline.incremental()` — full materialize → delta materialize → verify watermark advanced
- [x] T078 Test `commit_on_success=False` — watermark not persisted on failure
- [x] T079 Test watermark store scoping with custom `FileWatermarkStore(path=...)`

**Checkpoint**: User Story 4 complete — incremental pipeline works.

---

## Phase 7: User Story 5 — Filter Pushdown (Priority: P2) ✅

**Goal**: Automatic filter optimization with backend pushdown

### Implementation

- [x] T080 Complete `FilterHandler.split_pushdown_and_residual()` for SQL backend
- [x] T081 Complete `FilterHandler.split_pushdown_and_residual()` for Parquet backend
- [x] T082 Implement `_raw_filters_to_expression()` — convert filters to Arrow dataset filter expression
- [x] T083 Integrate filter splitting into `DataGateway.load()` — pushdown goes to backend, residual stays
- [x] T084 Add `strict_filter_validation` flag — deep validation of filter depth, condition count, IN-filter count

### Tests

- [x] T085 Test SQL pushdown — verify WHERE clause generation
- [x] T086 Test parquet partition pruning — verify only relevant partitions scanned
- [x] T087 Test combined pushdown+residual filters — result correctness
- [x] T088 Test strict_filter_validation — reject invalid filters

**Checkpoint**: User Story 5 complete — filter pushdown optimizes all backends.

---

## Phase 8: User Story 6 — Pipeline Enrichment (Priority: P3) ✅

**Goal**: Optional transform step between load and write

### Implementation

- [x] T089 Define `FrameEnricher` protocol in `enrichment/async_enricher.py`
- [x] T090 Create `AttachmentSpec` in `enrichment/specs.py`
- [x] T091 Integrate enricher into `SinkPipeline.write()` — call `_maybe_enrich_sync()` before sink write
- [x] T092 Create `HybridDataset` in `dataset/hybrid.py` — combine historical (parquet) + live (SQL) sources

### Tests

- [x] T093 Test enricher protocol — custom enricher adds column, verify output
- [x] T094 Test pipeline without enricher — passthrough, no extra columns
- [x] T095 Test `HybridDataset` — combined historical + live load

**Checkpoint**: User Story 6 complete — enrichment and hybrid sources work.

---

## Phase 9: Security & Observability (Cross-Cutting)

**Purpose**: Security hardening, diagnostics, resource lifecycle

- [x] T096 Add `diagnostics=True` support to `DataGateway` — log plan, partition, client summaries
- [x] T097 Add `WorkerSqlConfig` — minimal picklable config for distributed workers
- [x] T098 Add DSN serialization warning when `worker_connection_env_var` is not set
- [x] T099 Implement null-byte detection in `SecureResource.__init__`
- [x] T100 Add destructor warning for unclosed `ParquetDataResource`
- [x] T101 Create security regression test suite (8 tests) in `tests/security/test_regressions.py`
- [x] T102 Add `Makefile` with dev install, test, lint, type-check targets
- [x] T103 Add `examples/data_incremental_loading.py` — end-to-end incremental demo
- [x] T104 Add `notebooks/22_incremental_pipeline.ipynb` — interactive walkthrough
- [x] T105 Fix resource leak: add `pipeline.parquet_sink.close()` to example scripts

---

## Phase 10: Documentation & Polish

- [x] T106 Create `docs/SDD.md` — legacy architecture document (v1.0.0 format)
- [x] T107 Create this spec-kit SDD — `specs/boti-data-etl/` (constitution + spec + plan + tasks)
- [x] T108 Create `AGENTS.md` — AI agent guidance for the codebase
- [x] T109 Create `README.md` with install, usage, examples

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Foundation) → Phase 2 (Filters)
    → Phase 3 (SQL Load — US1)
        → Phase 4 (Parquet Load — US2)
            → Phase 5 (Materialize — US3)
                → Phase 6 (Incremental — US4)
                → Phase 7 (Pushdown — US5)
            → Phase 8 (Enrichment — US6)
→ Phase 9 (Security/Observability)
→ Phase 10 (Docs/Polish)
```

- Phases 3 and 4 can run in parallel after Phase 2 completes
- Phases 6 and 7 depend on Phase 5, but can run in parallel with each other
- Phase 8 depends on Phase 5 only

### Parallel Opportunities

| Task IDs | Rationale |
|----------|-----------|
| T004, T005, T006 | Config models — no dependencies |
| T010, T011 | Filter components — independent files |
| T013, T014 | Sync and async resources — independent |
| T024, T025 | Facade and frame strategies — independent |
| T056–T061 | Each sink test — independent file targets |
| T096–T101 | Security and diagnostics — independent concerns |

---

## Implementation Strategy

### Current Status

The framework is fully implemented at v1.0.1 with 441 passing tests. This task list documents the build order from scratch and serves as:
- An audit trail of what exists
- A reference for incremental feature planning
- A guide for new contributors to understand the dependency graph
