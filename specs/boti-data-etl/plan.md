# Implementation Plan: boti-data ETL Framework

**Branch**: `main` | **Date**: 2026-06-23 | **Spec**: `specs/boti-data-etl/spec.md`

**Input**: Feature specification from `specs/boti-data-etl/spec.md`

---

## Summary

boti-data is a Python ETL framework that loads data from relational databases (SQL) and Parquet datasets into Dask DataFrames (with pandas/Arrow/Polars alternatives), and materializes them to partitioned columnar storage with optional incremental watermark tracking. The framework follows a three-layer architecture: Facade (`DataHelper`) → Gateway (`DataGateway`) → Backend Resource (`SqlDatabaseResource`, `ParquetDataResource`), with a separate Pipeline layer for write orchestration.

---

## Technical Context

**Language/Version**: Python 3.13+

**Primary Dependencies**: Dask (2026.3.0+), SQLAlchemy (2.0+), pandas (3.0+), PyArrow, Polars (1.29+), pydantic, `boti` (core utilities), `boti-dask` (Dask runtime helpers)

**Storage**:
- SQL: SQLite (dev/test), MySQL/PgSQL (production) via SQLAlchemy dialects
- Parquet: Local filesystem (dev), S3/GCS via `filesystem_profile` (production)
- Watermarks: JSON files (local) — `FileWatermarkStore`

**Testing**: pytest with pytest-asyncio (`asyncio_mode = "auto"`), SQLite in temp directories, real parquet round-trips

**Target Platform**: Linux/macOS server, Python 3.13+

**Project Type**: Python library

**Performance Goals**:
- Auto-detection threshold: probe up to 10K rows (SQL) or 4 files / 32MB (parquet), then switch from dask to pandas for small datasets
- Partitioned SQL loading: N partitions → N concurrent queries via `ThreadPoolExecutor`
- Chunked IN loading: 900 values per chunk

**Constraints**:
- Read-only SQL by default — no DDL/DML via framework
- `connection_url` never logged; warning if serialized without `worker_connection_env_var`
- All file paths validated against `allowed_paths` (project root + temp)
- Pushdown filters must not change semantics vs. in-memory application

**Scale/Scope**: 1K–10M+ rows per table; single-node Dask (default) to multi-worker Dask cluster (distributed)

---

## Constitution Check

| Gate | Status | Notes |
|------|--------|-------|
| G1 — Constitution Check | ✅ Pass | All core principles followed. No unnecessary complexity. |
| G2 — Specification Completeness | ✅ Pass | 6 user stories, 20 FRs, 6 SCs, acceptance scenarios documented. |
| G3 — Test Gate | ✅ Pass | 441 tests exist across 18 test files. All fail-write-pass cycle verified. |
| G4 — Review Gate | ⏳ — | Applied at PR/merge time. |

---

## Project Structure

### Documentation (this feature)

```text
specs/boti-data-etl/
├── constitution.md      # Core principles and governance
├── spec.md              # Feature specification (this artifact)
├── plan.md              # Implementation plan (this file)
└── tasks.md             # Task breakdown

docs/
├── SDD.md               # Legacy architecture document (v1.0.0 format)
└── ...
```

### Source Code

```text
src/boti_data/
├── __init__.py                    # Public API — re-exports all key classes
├── helper.py                      # DataHelper facade + _EngineBoundHelper
├── parquet_reader.py              # ParquetReader (parquet-specialized DataHelper)
├── schema.py                      # Schema validation, dtype normalization
├── connection_catalog.py          # Named SQL/filesystem profiles
├── field_map.py                   # DB → semantic name translation
├── joins.py                       # indexed_left_join, left_join_frames
├── arrow_schema.py                # ArrowSchema utility
├── gateway/
│   ├── core.py                    # DataGateway — main orchestration engine
│   ├── requests.py                # Request models (SqlLoadRequest, ParquetLoadRequest)
│   ├── loaders.py                 # load_sql, load_parquet, build_backend_resource
│   ├── frame_strategies.py        # FrameStrategy variants (Dask, Pandas, Arrow, Polars)
│   ├── normalization.py           # Filter normalization, control key splitting
│   ├── arrow_adapters.py          # Arrow-native sort/dedup/groupby
│   └── sql_guard.py               # Raw SQL read-only validation
├── db/
│   ├── sql_config.py              # SqlDatabaseConfig, WorkerSqlConfig
│   ├── sql_resource.py            # SqlDatabaseResource (sync), AsyncSqlDatabaseResource
│   ├── sql_engine.py              # Engine creation helpers
│   ├── engine_registry.py         # EngineRegistry (singleton cache)
│   ├── sql_readonly.py            # ReadOnlySession wrappers
│   ├── sql_model_builder.py       # SqlAlchemyModelBuilder
│   ├── sql_model_registry.py      # SqlModelRegistry
│   ├── partitioned_loader.py      # SqlPartitionedLoader
│   ├── partitioned_planner.py     # SqlPartitionPlanner
│   ├── partitioned_execution.py   # SqlPartitionExecutor
│   ├── partitioned_types.py       # Partition request/plan models
│   └── arrow_schema_mapper.py     # Arrow ↔ SQLAlchemy mapping
├── parquet/
│   └── resource.py                # ParquetDataConfig, ParquetDataResource
├── pipelines/
│   ├── base.py                    # SinkPipeline, ParquetPipeline
│   ├── sinks.py                   # ParquetSink, CsvSink, JsonlSink
│   └── registry.py                # SinkRegistry
├── watermark/
│   ├── store.py                   # WatermarkStore protocol, FileWatermarkStore
│   └── incremental.py             # IncrementalResult, advance_watermark
├── filters/
│   ├── expressions.py             # Expr, And, Or, Not, ColOp
│   ├── handler.py                 # FilterHandler
│   ├── arrow_kernels.py           # Arrow-native filter kernels
│   └── utils.py                   # Filter utilities
├── enrichment/
│   ├── async_enricher.py          # FrameEnricher protocol
│   └── specs.py                   # AttachmentSpec
├── dataset/
│   └── hybrid.py                  # HybridDataset
└── datacube/
    ├── resource.py                # DatacubeResource
    └── contract.py                # DatacubeConfig, DatacubeContract

tests/
├── conftest.py
├── data/
│   ├── test_helper.py             # 912 lines — DataHelper, load modes, field_map
│   ├── test_pipelines.py          # 389 lines — ParquetPipeline, sinks, enricher
│   ├── test_watermark.py          # 661 lines — watermark store, incremental, delta
│   ├── test_facade.py             # Gateway facade operations
│   ├── test_filters.py            # Filter creation, combination, pushdown
│   ├── test_joins.py              # Join primitives
│   ├── test_parquet_resource.py   # File discovery, filtering, loading
│   ├── test_partitioned_sql_loader.py
│   ├── test_sql_model_builder.py
│   ├── test_enrichment.py
│   ├── test_hybrid_dataset.py
│   ├── test_frame_strategies.py
│   ├── test_arrow_adapters.py
│   ├── test_field_map_gateway.py
│   └── test_sink_registry.py
└── security/
    └── test_regressions.py        # 8 security regression tests

examples/
├── data_facade_db.py
├── data_facade_distributed.py
├── data_helper_distributed.py
└── data_incremental_loading.py

notebooks/
└── 22_incremental_pipeline.ipynb
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     User Code / App                      │
├───────────────────────┬──────────────────────────────────┤
│     DataHelper        │     ParquetPipeline              │
│     (facade)          │     (or SinkPipeline)            │
├──────────┬────────────┴──────────┬───────────────────────┤
│         DataGateway              │                        │
│  (structured / configured mode)  │                        │
│  - return type resolution        │                        │
│  - execution mode resolution     │                        │
│  - chunked IN loading            │                        │
│  - semi-join                     │                        │
│  - field_map translation         │                        │
│  - filter normalization          │                        │
├──────┬──────┬────────────────────┘                        │
│ SQL  │Parq  │Datacube                                    │
│ DB   │      │                                            │
├──────┤      │                                            │
│SqlDB │Parq  │Datacube                                    │
│Resrc │Resrc │Resource                                    │
└──────┴──────┴────────────────────────────────────────────┘
```

### Data Flow (Full ETL)

```
1. User calls pipeline.materialize(reload=True)
2. SinkPipeline.write() → forces dask/lazy execution mode
3. DataGateway.load() → builds load request, resolves execution plan
4. SqlPartitionedLoader → partitioned SQL execution → dd.DataFrame
5. Optional: FrameEnricher transforms data
6. ParquetSink.write() → to_dask_frame → prepare_partitioned_frame → ddf.to_parquet
7. If reload: ParquetReader.from_parquet() → reads back from disk
8. If incremental: advance_watermark() → store.write()
```

### Incremental Flow

```
Run 1: No watermark → full load → advance_watermark → store.write(max_date)
Run 2: Watermark exists → filter: {date_field__gt: watermark} → load delta
       → 0 rows: skip write, watermark unchanged
       → N rows: write delta → advance_watermark → store.write(new_max)
```

---

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Four backend types (sql, parquet, datacube, hybrid) | Each has distinct access patterns that do not unify under a single abstraction | A single `Resource` base class would leak backend-specific config into the gateway |
| Separate gateway and pipeline layers | Read/write separation principle; gateway handles load planning, pipeline handles write orchestration | Merging them would couple query planning to write concerns |
| Filter pushdown with two-phase evaluation | Performance-critical for large datasets; storage-layer filtering reduces data movement | In-memory-only filters would force full table scans on every load |

---

## Key Technical Decisions

1. **Dask-first**: Default return type is `dd.DataFrame`. This enables lazy evaluation so filter pushdown and column projection happen before data movement.

2. **`field_map` direction is DB → semantic**: Filters and `fieldnames` use semantic names; translation to internal DB column names happens inside the gateway. This keeps the public API stable when DB schemas change.

3. **`query_only` affects engine identity**: `SqlDatabaseConfig.query_only=True` creates a different engine than `query_only=False`. This is intentional — read-only engines can be cached and shared more aggressively.

4. **Watermark store scoping**: `FileWatermarkStore()` default path is `.watermarks.json` in CWD. For test isolation, always pass an explicit `path=` per run. This was a root cause of the incremental pipeline bug (fixed).

5. **`from_parquet()` skip on empty writes**: When incremental load produces 0 rows, `result.files` is empty, and the reload step is skipped entirely (rather than reading an empty directory and catching the warning).
