# Software Design Document — boti-data ETL Framework

**Version:** 1.0.1  
**Author:** Luis Valverde  
**Last Updated:** 2026-06-23

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Overview](#2-system-overview)
3. [Architecture](#3-architecture)
4. [Component Design](#4-component-design)
5. [Data Flow](#5-data-flow)
6. [Incremental Loading](#6-incremental-loading)
7. [Configuration Model](#7-configuration-model)
8. [Security Design](#8-security-design)
9. [Error Handling](#9-error-handling)
10. [Testing Strategy](#10-testing-strategy)
11. [Glossary](#11-glossary)

---

## 1. Introduction

### 1.1 Purpose

boti-data is a Python framework for building Extract-Transform-Load (ETL) pipelines
that source data from relational databases (SQL) and land it into columnar storage
(Parquet), optionally with incremental watermark tracking. It is designed around
Dask DataFrames as the primary in-memory representation, enabling lazy evaluation
and distributed execution for large datasets.

### 1.2 Scope

This document covers the architecture, component design, data flows, and key
design decisions of the boti-data package (`src/boti_data/`). It describes:

- The public API surface (`DataHelper`, `ParquetPipeline`, etc.)
- The core orchestration engine (`DataGateway`)
- The backend resources (SQL, Parquet, Datacube)
- The pipeline/sink abstraction for materialization
- The incremental loading / watermark tracking subsystem
- Configuration models and security boundaries

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|---|---|
| Dask | Parallel computing library for Python; provides `dask.dataframe` (lazy parallel DataFrame) |
| DSN | Data Source Name — connection string for a database |
| ETL | Extract, Transform, Load |
| FrameResult | A union type: `pd.DataFrame \| dd.DataFrame \| pa.Table \| pl.DataFrame` |
| Gateway | Core orchestration layer (`DataGateway`) |
| Hive Partitioning | Directory-based partitioning: `col=value/` |
| Pushdown | Applying a filter at the storage layer before data is loaded into memory |
| Watermark | A persisted value tracking the last successfully loaded record boundary |
| SDD | Software Design Document |

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Code / Application                      │
└──────────┬──────────────────────────────┬──────────────────────────┘
           │                              │
           ▼                              ▼
    ┌──────────────┐           ┌──────────────────────┐
    │  DataHelper  │           │   ParquetPipeline    │
    │  (facade)    │           │  (or SinkPipeline)   │
    └──────┬───────┘           └──────────┬───────────┘
           │                              │
           ▼                              ▼
    ┌──────────────────────────────────────────────┐
    │                DataGateway                    │
    │  (structured mode / configured mode)          │
    │  - return type resolution (auto/dask/pandas)  │
    │  - execution mode resolution (lazy/eager)     │
    │  - chunked IN loading                         │
    │  - semi-join (distributed)                    │
    │  - field_map translation                      │
    │  - filter normalization                       │
    └──┬──────────┬──────────┬──────────────────────┘
       │          │          │
       ▼          ▼          ▼
 ┌──────────┐ ┌────────┐ ┌────────────┐
 │ SQL DB   │ │ Parquet│ │ Datacube   │
 │ Backend  │ │ Backend│ │ Backend    │
 ├──────────┤ ├────────┤ ├────────────┤
 │SqlDBResrc│ │ParqResrc│ │DatacubeRes│
 │SqlPartLdr│ │PReader │ │            │
 │SqlModelBd│ │Filters │ │            │
 └──────────┘ └────────┘ └────────────┘
       │
       ▼
 ┌───────────────────┐
 │  EngineRegistry   │
 │  (singleton cache)│
 └───────────────────┘
```

### 2.2 Write Path

```
 ┌──────────┐    ┌──────────────┐    ┌──────────────┐
 │ DataHelp/│───▶│ SinkPipeline  │───▶│ ParquetSink  │
 │ HybridDs │    │ (or ParqPipe)│    │ CsvSink      │
 └──────────┘    └──────────────┘    │ JsonlSink    │
                      │              └──────────────┘
                      │ optional
                      ▼
               ┌──────────────┐
               │ FrameEnricher│
               └──────────────┘
```

### 2.3 Key Architectural Principles

1. **Dask-first**: All data loading defaults to Dask DataFrames for lazy,
   distributed execution. Eager pandas returns are opt-in (via `as_pandas=True`
   or auto-detection for small results).

2. **Two gateway modes**: "Structured mode" (pass `statement` + `model`
   explicitly) vs. "Configured mode" (set `table=` at construction; runtime
   kwargs become filters).

3. **Read/write separation**: `DataGateway` handles only loading (read).
   `SinkPipeline` / `ParquetSink` handle materialization (write).

4. **Pushdown filtering**: Filters are split into pushdown (applied at the
   storage layer) and residual (applied in-memory after load).

5. **Async-first**: All major operations have sync, async, and `*_sync()`
   convenience wrappers.

---

## 3. Architecture

### 3.1 Package Structure

```
boti-data/
├── pyproject.toml
├── src/boti_data/
│   ├── __init__.py           # Public API — re-exports all key classes
│   ├── helper.py             # DataHelper facade + _EngineBoundHelper
│   ├── parquet_reader.py     # ParquetReader (parquet-specialized DataHelper)
│   ├── schema.py             # Schema validation, dtype normalization, alignment
│   ├── connection_catalog.py # Catalog of named SQL/filesystem profiles
│   ├── field_map.py          # DB→semantic name translation
│   ├── joins.py              # indexed_left_join, left_join_frames
│   ├── arrow_schema.py       # ArrowSchema utility
│   ├── gateway/
│   │   ├── core.py           # DataGateway — main orchestration engine
│   │   ├── requests.py       # Request models (SqlLoadRequest, ParquetLoadRequest)
│   │   ├── loaders.py        # load_sql, load_parquet, build_backend_resource
│   │   ├── frame_strategies.py  # FrameStrategy variants (Dask, Pandas, Arrow, Polars)
│   │   ├── normalization.py  # Filter normalization, control key splitting
│   │   ├── arrow_adapters.py # Arrow-native sort/dedup/groupby
│   │   └── sql_guard.py      # Raw SQL read-only validation
│   ├── db/
│   │   ├── sql_config.py     # SqlDatabaseConfig, WorkerSqlConfig
│   │   ├── sql_resource.py   # SqlDatabaseResource (sync), AsyncSqlDatabaseResource
│   │   ├── sql_engine.py     # Engine creation helpers
│   │   ├── engine_registry.py # EngineRegistry (singleton cache)
│   │   ├── sql_readonly.py   # ReadOnlySession wrappers
│   │   ├── sql_model_builder.py # SqlAlchemyModelBuilder
│   │   ├── sql_model_registry.py # SqlModelRegistry
│   │   ├── partitioned_loader.py # SqlPartitionedLoader
│   │   ├── partitioned_planner.py # SqlPartitionPlanner
│   │   ├── partitioned_execution.py # SqlPartitionExecutor
│   │   ├── partitioned_types.py # Partition request/plan models
│   │   └── arrow_schema_mapper.py # Arrow<->SQLAlchemy mapping
│   ├── parquet/
│   │   ├── resource.py       # ParquetDataConfig, ParquetDataResource
│   ├── pipelines/
│   │   ├── base.py           # SinkPipeline, ParquetPipeline
│   │   ├── sinks.py          # ParquetSink, CsvSink, JsonlSink
│   │   └── registry.py       # SinkRegistry
│   ├── watermark/
│   │   ├── store.py          # WatermarkStore protocol, FileWatermarkStore
│   │   └── incremental.py    # IncrementalResult, advance_watermark
│   ├── filters/
│   │   ├── expressions.py    # Expr, And, Or, Not, ColOp
│   │   ├── handler.py        # FilterHandler
│   │   ├── arrow_kernels.py  # Arrow-native filter kernels
│   │   └── utils.py          # Filter utilities
│   ├── enrichment/
│   │   ├── async_enricher.py # FrameEnricher protocol
│   │   └── specs.py          # AttachmentSpec
│   ├── dataset/
│   │   └── hybrid.py         # HybridDataset
│   └── datacube/
│       ├── resource.py       # DatacubeResource
│       └── contract.py       # DatacubeConfig, DatacubeContract
└── tests/
    ├── conftest.py
    ├── data/
    │   ├── test_helper.py
    │   ├── test_pipelines.py
    │   ├── test_watermark.py
    │   ├── test_facade.py
    │   ├── test_filters.py
    │   ├── test_joins.py
    │   ├── test_parquet_resource.py
    │   └── ... (18 test files)
    └── security/
        └── test_regressions.py
```

### 3.2 Module Dependency Graph

```
__init__.py  (public API surface)
    │
    ├── helper.py  ─────────────────────────────────────┐
    │   └── gateway/core.py (DataGateway)                │
    │       ├── gateway/requests.py (models)             │
    │       ├── gateway/loaders.py (load implementations)│
    │       │   ├── db/ (SqlDatabaseResource, etc.)      │
    │       │   └── parquet/resource.py (ParquetDataRes) │
    │       ├── gateway/normalization.py                 │
    │       └── gateway/frame_strategies.py              │
    │                                                    │
    ├── parquet_reader.py  ──────────────────────────────┤
    │   └── helper.py  (inherits DataHelper)             │
    │                                                    │
    ├── pipelines/base.py  ──────────────────────────────┤
    │   ├── helper.py  (source)                          │
    │   └── pipelines/sinks.py (destination)             │
    │       └── parquet/resource.py (ParquetDataResource)│
    │                                                    │
    ├── watermark/  ─────────────────────────────────────┘
    │   ├── store.py (FileWatermarkStore)
    │   └── incremental.py (advance_watermark)
    │
    ├── filters/  ───────────────────────────────────────┐
    │   └── handler.py (FilterHandler)                   │
    │       └── filters/expressions.py                   │
    │                                                    │
    ├── field_map.py ────────────────────────────────────┤
    └── joins.py  ───────────────────────────────────────┘
```

### 3.3 Class Hierarchy

```
boti.core.ResourceConfig (Pydantic BaseModel)
  ├── SqlDatabaseConfig
  │   └── connection_url: SecretStr, query_only, pool_size, etc.
  │
  ├── ParquetDataConfig
  │   └── parquet_storage_path, partition_on, filesystem_profile, etc.
  │
  └── DatacubeConfig
      └── contract: DatacubeContract

boti.core.SecureResource
  ├── ParquetDataResource
  │   ├── load_files() → dd.DataFrame
  │   ├── load_filtered() → dd.DataFrame
  │   ├── load_arrow() → pa.Table
  │   └── _resolve_files_to_load() → file discovery
  │
  ├── SqlDatabaseResource
  │   └── engine, session properties
  │
  └── AsyncSqlDatabaseResource

DataGateway
  ├── config: BackendConfig
  ├── backend: BackendName
  ├── resource: BackendResource | None
  ├── load(**options) → FrameResult
  ├── aload(**options) → FrameResult
  └── from_config(), from_backend() — constructors

DataHelper (facade over DataGateway)
  ├── load(), aload(), load_incremental()
  ├── .dask → _EngineBoundHelper
  ├── .pandas → _EngineBoundHelper
  └── .polars → _EngineBoundHelper

ParquetReader(DataHelper)
  ├── config → ParquetDataConfig
  ├── resource → ParquetDataResource
  └── parquet_storage_path

SinkPipeline
  ├── source: PipelineSource
  ├── sink: PipelineSink
  ├── write() → loads + writes
  └── load() → delegates to source

ParquetPipeline(SinkPipeline)
  ├── parquet_sink: ParquetSink
  ├── reader: ParquetReader
  ├── from_parquet() → reload
  ├── to_parquet() → write
  ├── materialize() → write + optional reload
  └── incremental() → class factory for incremental mode

PipelineSink (Protocol)
  ├── ParquetSink
  ├── CsvSink
  └── JsonlSink

WatermarkStore (Protocol)
  └── FileWatermarkStore (JSON-file-backed)
```

### 3.4 Type System

```python
BackendName = Literal["sqlalchemy", "parquet", "datacube"]
BackendConfig = Union[SqlDatabaseConfig, ParquetDataConfig, DatacubeConfig]
BackendResource = Union[SqlDatabaseResource, ParquetDataResource, DatacubeResource]
ReturnType = Literal["pandas", "arrow", "dask", "polars", "auto"]
ResolvedReturnType = Literal["pandas", "arrow", "dask", "polars"]
ExecutionMode = Literal["eager", "lazy", "auto"]
ResolvedExecutionMode = Literal["eager", "lazy"]
FrameResult = Union[pd.DataFrame, dd.DataFrame, pa.Table, pl.DataFrame]
PipelineSource = Union[DataHelper, HybridDataset]
ParquetDestination = Union[ParquetReader, ParquetDataConfig, Mapping[str, Any]]
```

---

## 4. Component Design

### 4.1 DataHelper (`helper.py`)

**Purpose:** Public facade over `DataGateway`. Entry point for all data loading
operations.

**Key Interfaces:**

```python
class DataHelper:
    def __init__(self, config: DataGateway | BackendConfig | dict | None, **overrides)

    def load(self, **options) -> FrameResult
    def aload(self, **options) -> FrameResult
    def aload_sync(self, **options) -> FrameResult

    def preview(self, *, n=5, npartitions=1, **options)
    def load_period(self, dt_field, start, end, **kwargs)
    def semi_join(self, join_series, on, **kwargs)

    def load_incremental(self, *, watermark_field, watermark_source=None,
                         watermark_store=None, initial_value=None,
                         operator="gt", commit_on_success=True, **load_options)
        -> IncrementalResult

    @property
    def dask(self) -> _EngineBoundHelper     # return_type="dask", execution_mode="lazy"
    @property
    def pandas(self) -> _EngineBoundHelper   # return_type="pandas", execution_mode="eager"
    @property
    def polars(self) -> _EngineBoundHelper   # return_type="polars", execution_mode="eager"

    @staticmethod
    def left_join(left, right, *, join_schema_map, ...)

    def close(self)
```

**Design Notes:**
- Accepts multiple config formats: `DataGateway` instance, a config model
  (`SqlDatabaseConfig` etc.), a dict (delegated to `DataGateway.from_config()`),
  or raw kwargs.
- `_EngineBoundHelper` enforces compatibility — e.g., `helper.pandas` rejects
  `return_type="dask"` with a clear error.
- `aload_sync()` bridges async loads into synchronous code; raises if an event
  loop is already running (notebooks must use `await`).

### 4.2 DataGateway (`gateway/core.py`)

**Purpose:** Core orchestration engine. Resolves execution plans, dispatches to
backend-specific load implementations, manages return-type coercion.

**Key Interfaces:**

```python
class DataGateway:
    def __init__(self, config: BackendConfig, *, table=None, field_map=None,
                 sticky_filters=None, df_params=None, df_options=None, ...)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **overrides)
    @classmethod
    def from_backend(cls, backend: BackendName, **config)

    def load(self, **options) -> FrameResult
    def aload(self, **options) -> FrameResult
    def preview(self, *, n=5, npartitions=1, **options)
    def apreview(self, *, n=5, npartitions=1, **options)

    def load_period(self, dt_field, start, end, **kwargs)
    def aload_period(self, dt_field, start, end, **kwargs)

    def semi_join(self, join_series, on, **kwargs)
    def asemi_join(self, join_series, on, **kwargs)
```

**Execution Plan Resolution (`_resolve_execution_plan`):**

1. `_resolve_requested_return_type()`: reads `return_type` from options, or
   falls back to configured default, or `"dask"`.
2. `_resolve_requested_execution_mode()`: reads `execution_mode` from options,
   or falls back to configured default, or `"auto"`.
3. `_resolve_execution_plan()`: if `"auto"`:
   - **SQL**: probes row count up to threshold (10,000). Small → pandas, large → dask.
   - **Parquet**: checks file count (<= 4) and total bytes (<= 32 MB). Small → pandas, large → dask.
4. `_execute_sync()` dispatches by backend:
   - **SQL (lazy)**: `load_sql_partitioned()` — Dask distributed
   - **SQL (eager)**: `load_sql()` — pandas
   - **Parquet**: `load_parquet()` — `dd.read_parquet()` or `ds.dataset()`
   - **Datacube**: `load_datacube()`

**Chunked IN Loading:**
- For queries with large `field__in=[...]` lists, splits into chunks of 900
  values, executes concurrently via `ThreadPoolExecutor`, concatenates results.

### 4.3 ParquetPipeline (`pipelines/base.py`)

**Purpose:** Materialize data from a source (SQL `DataHelper` or `HybridDataset`)
into a parquet dataset, with optional reload and incremental watermark tracking.

**Key Interfaces:**

```python
class ParquetPipeline(SinkPipeline):
    def __init__(self, source, destination, *, date_field=None, partition_on=("partition_date",))

    def to_parquet(self, *, write_index=False, overwrite=True, persist=False, **load_options) -> str
    def ato_parquet(self, ...) -> str
    def from_parquet(self, **options)
    def afrom_parquet(self, **options)

    def materialize(self, *, reload=False, reload_options=None, write_index=False,
                    overwrite=True, persist=False, **load_options)
        -> ParquetMaterializationResult

    async def amaterialize(self, ...) -> ParquetMaterializationResult

    @classmethod
    def incremental(cls, source, destination, *, watermark_field, watermark_store=None,
                    watermark_source=None, initial_value=None, date_field=None,
                    partition_on=("partition_date",))
        -> ParquetPipeline
```

**Materialization Flow:**

```
materialize(reload=True)
  │
  ├── _watermark_field is set?
  │   ├── Yes → _materialize_incremental()
  │   │   1. Read previous watermark from store
  │   │   2. Build incremental filter: {watermark_field__gt: previous}
  │   │   3. Merge with any explicit filters
  │   │   4. self.write() → self.load() + sink.write() → parquet
  │   │   5. If reload and files were written: self.from_parquet() → reload frame
  │   │   6. Update watermark from reloaded frame (or from source if no reload)
  │   │   7. Return ParquetMaterializationResult(path, frame)
  │   │
  │   └── No → _materialize_full()
  │       1. self.write() → full load + parquet write
  │       2. If reload: self.from_parquet()
  │       3. Return ParquetMaterializationResult(path, frame)
```

### 4.4 ParquetSink (`pipelines/sinks.py`)

**Purpose:** Write a frame to a partitioned parquet dataset.

```python
class ParquetSink:
    def __init__(self, destination, *, partition_on=("partition_date",))

    def write(self, frame, *, date_field=None, write_index=False,
              overwrite=True, persist=False)
        -> SinkWriteResult

    async def awrite(self, ...) -> SinkWriteResult
```

**Write Flow:**

```
sink.write(frame, date_field="event_date")
  │
  1. to_dask_frame(frame)       # Normalize: pd/polars/pa → dd.DataFrame
  2. prepare_partitioned_frame(ddf, partition_on, date_field)
     # If "partition_date" is missing, derive from date_field
     # via dd.to_datetime(...).dt.date.astype(str)
  3. If overwrite: fs.rm(target_path, recursive=True)
  4. ddf.to_parquet(path=target_path, partition_on=[...], filesystem=fs, engine="pyarrow")
  5. fs.glob(f"{target_path}/*") → collect written files
  6. Return SinkWriteResult(path, files)
```

### 4.5 ParquetDataResource (`parquet/resource.py`)

**Purpose:** Secure resource for discovering and loading parquet data.

```python
class ParquetDataResource(SecureResource):
    def load_files(self, filters=None, *, columns=None) -> dd.DataFrame
    def load_arrow(self, filters=None, *, columns=None) -> pa.Table
    def load_filtered(self, filters, *, columns=None) -> dd.DataFrame
    def load_filtered_arrow(self, filters, *, columns=None) -> pa.Table
    async def aload_files(self, ...) -> dd.DataFrame
    async def aload_filtered(self, ...) -> dd.DataFrame
```

**File Discovery (`_resolve_files_to_load`):**

```
_resolve_files_to_load()
  │
  ├── parquet_filename is set?
  │   ├── Yes → determine_recency() → [single file path]
  │   └── No → continue
  │
  ├── parquet_start_date AND parquet_end_date set?
  │   ├── Yes → _discover_partitioned_files()
  │   │   # Uses pyarrow.dataset with hive partitioning
  │   │   # or falls back to listing + range matching
  │   └── No → continue
  │
  └── _discover_all_files()
      # Scans the entire dataset via ds.dataset()
      # or returns [] if path doesn't exist
```

### 4.6 Watermark Subsystem (`watermark/`)

**Purpose:** Track the last successfully loaded boundary for incremental loads.

**Protocol:**

```python
class WatermarkStore(Protocol):
    def read(self, *, source: str) -> Any | None
    def write(self, *, source: str, value: Any) -> None
    def clear(self, *, source: str) -> None
```

**Implementation — FileWatermarkStore:**
- Thread-safe (`threading.Lock`)
- JSON file format: `{"source_name": watermark_value, ...}`
- Default path: `.watermarks.json` (relative to CWD)

**Key Functions:**

```python
def build_incremental_filters(*, watermark_field, watermark_value, operator="gt")
    # → {"updated_at__gt": watermark_value}

def advance_watermark(frame, *, watermark_field) -> Any | None
    # Returns max(watermark_field) — supports pd/dd/pl/pa frame types
```

### 4.7 SinkPipeline (`pipelines/base.py`)

**Purpose:** Generic orchestration layer that loads from a source and writes to
a sink. Intentionally sits _above_ `DataGateway` to keep the gateway focused on
reads.

```python
class SinkPipeline:
    def __init__(self, source, sink, *, sink_config=None, enricher=None, date_field=None)

    def write(self, *, write_index=False, overwrite=True, persist=False, **load_options)
        -> SinkWriteResult
    async def awrite(self, ...) -> SinkWriteResult

    def load(self, **options) -> FrameResult
    async def aload(self, **options) -> FrameResult
```

**Load Option Enforcement:**
- Forces `return_type="dask"` and `execution_mode="lazy"` — all materialization
  writes use Dask for lazy evaluation.
- Rejects `as_pandas=True` as incompatible with distributed writes.

### 4.8 Other Sinks

**CsvSink** (`pipelines/sinks.py:127`):
- Writes partitioned or flat CSV datasets via `ddf.to_csv()`
- Single partition column max
- Pattern: `part-*.csv`

**JsonlSink** (`pipelines/sinks.py:292`):
- Writes partitioned or flat JSONL datasets via `ddf.to_json(lines=True)`
- Single partition column max
- Pattern: `part-*.jsonl`

### 4.9 FilterHandler (`filters/handler.py`)

**Purpose:** Normalize and apply filter expressions with pushdown optimization.

```python
class FilterHandler:
    def __init__(self, backend, *, logger, debug=False)

    def split_pushdown_and_residual(self, filters)
        # Returns (pushdown_filters, residual_filters)
        # Pushdown: filters the backend can handle natively (e.g., SQL WHERE)
        # Residual: filters that must be applied in-memory

    def apply_filters(self, dataframe, *, filters)
```

---

## 5. Data Flow

### 5.1 Full ETL Materialization (SQL → Parquet)

```
User calls:
    pipeline = ParquetPipeline(source, destination)
    result = pipeline.materialize(reload=True, filters={"status": "active"})

Internal flow:
  ┌────────────────────────────────────────────────────────────┐
  │ 1. ParquetPipeline.materialize()                           │
  │    → _materialize_incremental() or _materialize_full()    │
  └──────────────────────┬─────────────────────────────────────┘
                         │
  ┌──────────────────────▼─────────────────────────────────────┐
  │ 2. SinkPipeline.write()                                    │
  │    a) _materialization_load_options() → force dask/lazy    │
  │    b) self.load(options)                                   │
  └──────────────────────┬─────────────────────────────────────┘
                         │
  ┌──────────────────────▼─────────────────────────────────────┐
  │ 3. DataGateway.load()                                      │
  │    a) Resolve return_type & execution_mode                 │
  │    b) Build SQL load request (structured/configured)       │
  │    c) If lazy: load_sql_partitioned() → dd.DataFrame       │
  │       If eager: load_sql() → pd.DataFrame                  │
  │    d) Apply field_map, column selection                    │
  └──────────────────────┬─────────────────────────────────────┘
                         │
  ┌──────────────────────▼─────────────────────────────────────┐
  │ 4. SqlPartitionedLoader                                    │
  │    a) SqlPartitionPlanner.plan_request()                   │
  │       → partition key, value ranges, partition SQL stmts   │
  │    b) SqlPartitionExecutor.load_plan()                     │
  │       → executes partitions, returns dd.DataFrame          │
  └──────────────────────┬─────────────────────────────────────┘
                         │
  ┌──────────────────────▼─────────────────────────────────────┐
  │ 5. SinkPipeline write (cont.)                              │
  │    a) _maybe_enrich_sync() — optional FrameEnricher        │
  │    b) self.sink.write(frame, date_field)                   │
  └──────────────────────┬─────────────────────────────────────┘
                         │
  ┌──────────────────────▼─────────────────────────────────────┐
  │ 6. ParquetSink.write()                                     │
  │    a) to_dask_frame(frame) → dd.DataFrame                  │
  │    b) prepare_partitioned_frame()                          │
  │       → derive partition_date from date_field if missing   │
  │    c) fs.rm(target_path) if overwrite                      │
  │    d) ddf.to_parquet(path, partition_on, filesystem)       │
  │    e) fs.glob() → collect written file paths               │
  │    f) Return SinkWriteResult(path, files)                  │
  └──────────────────────┬─────────────────────────────────────┘
                         │
  ┌──────────────────────▼─────────────────────────────────────┐
  │ 7. ParquetPipeline materialize (cont.)                     │
  │    If reload=True and files were written:                  │
  │    a) self.from_parquet() → ParquetReader.load()           │
  │       → load_parquet() → ParquetDataResource.load_files()  │
  │       → dd.read_parquet(files)                             │
  │    b) If frame has data: update watermark via store.write()│
  │    c) Return ParquetMaterializationResult(path, frame)     │
  └────────────────────────────────────────────────────────────┘
```

### 5.2 SQL Load Path Detail

**Eager** (`loaders.py:148`):
```
load_sql(resource, request)
  → _prepare_sql_statement(request)
  → resource.engine.connect()
  → conn.execute(statement, params) / pd.read_sql(statement, conn)
  → Results as pandas DataFrame or pyarrow Table
```

**Lazy/Partitioned** (`loaders.py:177`):
```
load_sql_partitioned(config, resource, request)
  → SqlPartitionedLoader(config, resource).load_request(request)
  → SqlPartitionPlanner.prepare_statement(request)
  → SqlPartitionPlanner.plan_request() → produces SqlPartitionPlan
  → SqlPartitionExecutor.load_plan() → executes partitions
  → Returns dd.DataFrame
```

The partition planner splits on a partition column (e.g., `id` range or date
range). Each partition becomes a separate SQL query executed as a Dask partition.

### 5.3 Parquet Load Path Detail

**Dask path** (`load_parquet` → `load_files`):
```
load_parquet(resource, request)
  → resource.load_files(filters=None, columns=None)
    → _resolve_files_to_load()
      → _discover_all_files()  # or _discover_partitioned_files()
        → ds.dataset(source_path, format="parquet")
          → [file1.parquet, file2.parquet, ...]
    → dd.read_parquet(files, filesystem=..., engine="pyarrow", ...)
```

**Arrow path** (`load_arrow`):
```
resource.load_arrow(filters=None, columns=None)
  → _resolve_files_to_load()
  → ds.dataset(files, filesystem, format="parquet")
  → _raw_filters_to_expression() → Arrow dataset filter
  → dataset.to_table(filter=expression, columns=columns)
  → pa.Table
```

---

## 6. Incremental Loading

### 6.1 Concept

Incremental loading tracks the maximum value of a designated watermark field
(typically a date or auto-incrementing ID) across materialization runs. On
subsequent runs, only rows with a watermark value _greater_ than the persisted
value are loaded.

### 6.2 Flow Diagram

```
Run 1 (no watermark):
  ┌─────────┐     ┌──────────────┐     ┌─────────────────┐
  │ SELECT * │────▶│ Write to     │────▶│ Reload from     │
  │ FROM tbl │     │ parquet      │     │ parquet         │
  └─────────┘     └──────────────┘     └────────┬────────┘
                                                │
                                                ▼
                                        ┌─────────────────┐
                                        │ advance_watermark│
                                        │ → max(event_date)│
                                        │ = 2026-05-05    │
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │ store.write(    │
                                        │  value=2026-05-05│
                                        └─────────────────┘

Run 2 (watermark exists):
  ┌──────────────────┐     ┌──────────────┐     ┌─────────────┐
  │ SELECT *          │────▶│ Write to     │────▶│ No files    │
  │ FROM tbl          │     │ parquet      │     │ written     │
  │ WHERE event_date  │     │ (0 rows)     │     │ (handled)   │
  │ > '2026-05-05'   │     └──────────────┘     └─────────────┘
  └──────────────────┘
  (0 rows returned, no files produced)

Run 3 (new data inserted):
  ┌──────────────────┐     ┌──────────────┐     ┌─────────────────┐
  │ SELECT *          │────▶│ Write to     │────▶│ advance_watermark│
  │ FROM tbl          │     │ parquet      │     │ → max = 2026-05-07│
  │ WHERE event_date  │     │ (2 rows)     │     └────────┬────────┘
  │ > '2026-05-05'   │     └──────────────┘              │
  └──────────────────┘                                    ▼
                                                   store.write(
                                                   value=2026-05-07)
```

### 6.3 Key Methods

**DataHelper.load_incremental()** (`helper.py:279`):
```python
def load_incremental(self, *, watermark_field, watermark_source=None,
                     watermark_store=None, initial_value=None,
                     operator="gt", commit_on_success=True, **load_options):
    # 1. Resolve source name and store
    # 2. Read previous watermark
    # 3. Build incremental filters
    # 4. Load data with filters applied
    # 5. Compute records_loaded
    # 6. Advance watermark → determine new max value
    # 7. If commit_on_success: persist new watermark
    # 8. Return IncrementalResult(...)
```

**ParquetPipeline._materialize_incremental()** (`base.py:430`):
```python
def _materialize_incremental(self, *, reload, reload_options, ...):
    # 1. Read previous watermark
    # 2. Build incremental filters
    # 3. self.write() → load from source + write to parquet
    # 4. If reload and result.files:
    #     a. self.from_parquet() → read back
    #     b. If frame has columns with data: update watermark
    # 5. If not reload: update watermark from source load
    # 6. Return ParquetMaterializationResult(path, frame)
```

### 6.4 Watermark Persistence

The default `FileWatermarkStore` persists to a JSON file:

```json
{
  "events_table": "2026-05-05 00:00:00+00:00",
  "events_pipeline": "2026-05-07 00:00:00+00:00"
}
```

---

## 7. Configuration Model

### 7.1 SqlDatabaseConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `connection_url` | `SecretStr` | required | Database DSN |
| `query_only` | `bool` | `True` | Read-only enforcement |
| `worker_connection_env_var` | `str \| None` | `None` | Env var for DSN on Dask workers |
| `pool_size` | `int` | `5` | SQLAlchemy pool size |
| `max_overflow` | `int` | `10` | Max overflow connections |
| `pool_timeout` | `int` | `30` | Pool connection timeout |
| `pool_recycle` | `int` | `1800` | Connection recycle time (seconds) |
| `pool_pre_ping` | `bool` | `True` | Connection health check |
| `poolclass` | `type[Pool]` | `QueuePool` | Pool implementation class |
| `connect_args` | `dict` | `{}` | Driver-specific connection args |
| `execution_options` | `dict` | `{}` | SQLAlchemy engine execution options |

Factory methods: `from_settings()`, `from_env()`, `from_env_prefix()`

### 7.2 ParquetDataConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `parquet_storage_path` | `str \| None` | `None` | Base path for parquet files |
| `parquet_filename` | `str \| None` | `None` | Single file name (not a directory) |
| `parquet_start_date` | `dt.date \| None` | `None` | Start of date range filter |
| `parquet_end_date` | `dt.date \| None` | `None` | End of date range filter |
| `parquet_max_age_minutes` | `int` | `0` | Max file age (0 = no limit) |
| `partition_on` | `list[str] \| None` | `None` | Hive partition columns |
| `filesystem_profile` | `str \| None` | `None` | Named profile for remote filesystems |

### 7.3 DataFrameParams

| Field | Type | Default | Description |
|---|---|---|---|
| `fieldnames` | `tuple[str, ...] \| None` | `None` | Semantic columns to SELECT |
| `column_names` | `list[str] \| None` | `None` | Final positional column rename |
| `chunk_size` | `int \| None` | `None` | Partition size for SQL loads |
| `index_col` | `str \| None` | `None` | Index column |
| `datetime_index` | `str \| None` | `None` | Datetime index column |
| `return_type` | `ReturnType` | `"dask"` | Output type |
| `execution_mode` | `ExecutionMode` | `"auto"` | Fetch strategy |

### 7.4 DataFrameOptions

| Field | Type | Default | Description |
|---|---|---|---|
| `sort_field` | `str \| None` | `None` | Sort by this column |
| `duplicate_expr` | `str \| list[str] \| None` | `None` | Column(s) for dedup |
| `duplicate_keep` | `Literal` | `"last"` | Which duplicate to keep |
| `group_by_expr` | `str \| list[str] \| None` | `None` | Group-by columns |
| `group_expr` | `str \| dict \| None` | `None` | Aggregation expression |

---

## 8. Security Design

### 8.1 Read-Only Enforcement

- `SqlDatabaseConfig.query_only` (default `True`) restricts the engine to
  read-only connections. When `True`, the gateway raises if write operations
  are attempted.
- `RawSqlPolicy` controls raw SQL statements:
  - `"disabled"`: raw SQL is forbidden entirely.
  - `"readonly_opt_in"`: raw SQL is allowed only if explicitly opted in via
    `allow_raw_sql=True` and validated as read-only by `sql_guard.py`.

### 8.2 Path Traversal Protection

- `SecureResource` validates that all file paths are within configured
  `allowed_paths` (defaults to `project_root` and system temp directory).
- Null-byte injection detected at resource construction and rejected.
- All parquet storage paths validated through `get_secure_path()`.

### 8.3 Credential Safety

- `connection_url` stored as `pydantic.SecretStr` — never logged in plaintext.
- DSN serialization warning: when `worker_connection_env_var` is not set,
  a `UserWarning` warns that the raw DSN will be pickled to workers.
- `WorkerSqlConfig` provides a minimal picklable config variant that avoids
  serializing credentials.

---

## 9. Error Handling

### 9.1 Error Categories

| Category | Examples | Handling |
|---|---|---|
| Configuration | Missing `parquet_storage_path`, invalid `connection_url` | Pydantic validation errors at construction |
| Security | Path traversal, null-byte, raw SQL disabled | `PermissionError`, `ValueError` |
| Backend | DB connection failure, parquet file not found | `SQLAlchemyError`, `FileNotFoundError` → wrapped in `RuntimeError` |
| API misuse | Wrong `return_type`, conflicting filters | `TypeError`, `ValueError` with descriptive messages |
| Resource | Parquet resource garbage collected without close | `__del__` logs warning |
| Incremental | No watermark file, empty frame | Graceful: returns `None` for frame, advances watermark only when data present |

### 9.2 Key Error Handling Patterns

- **Filter validation**: `strict_filter_validation` flag enables deep validation
  of filter depth, condition count, and IN-filter value count.
- **Empty datasets**: `_empty_ddf()` creates an empty Dask DataFrame with
  correct schema if no parquet files exist.
- **Resource cleanup**: `close()` methods on all resources; context manager
  support (`__enter__`/`__exit__`); destructor warning for unclosed resources.

---

## 10. Testing Strategy

### 10.1 Test Infrastructure

- **Framework**: pytest with pytest-asyncio (`asyncio_mode = "auto"`)
- **Database**: SQLite in temp directories for full integration tests
- **Parquet**: Real parquet files written to temp directories and read back
- **Fixtures**: `temp_project_root`, `temp_log_dir` in `conftest.py`

### 10.2 Test Coverage

| Test File | Lines | Coverage |
|---|---|---|
| `test_helper.py` | 912 | DataHelper construction, load modes, field_map, engine-bound views, semi_join, load_period, left_join, error cases |
| `test_watermark.py` | 661 | FileWatermarkStore, build_incremental_filters, advance_watermark (all frame types), DataHelper.load_incremental, ParquetPipeline.incremental, delta loads, commit behavior |
| `test_pipelines.py` | 389 | ParquetPipeline (DataHelper + HybridDataset), materialize/reload, to_parquet, async, sink validation; SinkPipeline with CSV, JSONL, registry, enricher |
| `test_facade.py` | — | DataGateway facade operations |
| `test_filters.py` | — | Filter creation, combination, pushdown |
| `test_joins.py` | — | Join primitives |
| `test_parquet_resource.py` | — | File discovery, filtering, loading |
| `test_partitioned_sql_loader.py` | — | SQL partitioned execution |
| `test_sql_model_builder.py` | — | Model reflection and building |
| `test_enrichment.py` | — | FrameEnricher protocol |
| `test_hybrid_dataset.py` | — | HybridDataset (historical + live) |
| `test_frame_strategies.py` | — | Return type coercion |
| `test_arrow_adapters.py` | — | Arrow-native utilities |
| `test_field_map_gateway.py` | — | FieldMap integration |
| `test_sink_registry.py` | — | Named sink factories |
| `test_security/regressions.py` | — | Security regression tests |

### 10.3 Test Commands

```bash
uv run pytest tests/ -m "not security_regression"
uv run pytest tests/ -m security_regression
uv run pytest tests/data/test_pipelines.py -x -v
uv run pytest tests/data/test_watermark.py -x -v
```

### 10.4 Test Patterns

- Full ETL flows: seed SQLite → create DataHelper → load incremental →
  verify watermark commit → add new data → verify delta load
- Async tests use `@pytest.mark.asyncio` with `async with` patterns
- Security tests marked with `@pytest.mark.security_regression`
- Each test creates temporary directories that are cleaned up after

---

## 11. Glossary

| Term | Definition |
|---|---|
| **Backend** | A data source or sink type: `"sqlalchemy"`, `"parquet"`, or `"datacube"` |
| **Configured Mode** | Gateway mode where `table=` is set at construction; runtime kwargs become filters |
| **Dask** | Parallel computing library; provides lazy, distributed DataFrames |
| **DSN** | Data Source Name — connection URL for a database |
| **Enricher** | Optional transform step between load and write in a pipeline |
| **FrameResult** | Union type for all supported DataFrame representations |
| **Gateway** | `DataGateway` — the core orchestration engine |
| **Hive Partitioning** | Directory-based partitioning scheme: `column=value/file.parquet` |
| **Incremental** | Loading only new/changed data since the last run |
| **Materialize** | The act of loading from a source and writing to a sink (parquet, CSV, etc.) |
| **Partition** | A division of data (SQL: `WHERE id BETWEEN X AND Y`; Parquet: `col=val/`) |
| **Pushdown** | Applying a filter at the storage layer before data enters memory |
| **Residual** | A filter applied in-memory after data is loaded |
| **Structured Mode** | Gateway mode where `statement` + `model` are passed explicitly |
| **Watermark** | A persisted value indicating the last successfully loaded record boundary |
