# PyArrow Integration Proposal — Enhancing Boti Performance

## Executive Summary

Boti currently uses PyArrow **only for filesystem I/O and parquet dataset discovery**. All in-memory data flows through pandas/Dask, missing significant performance opportunities. This proposal outlines a phased integration strategy to leverage PyArrow's **zero-copy reads**, **vectorized compute kernels**, and **batch coercion** across critical performance paths.

**Expected impact:** 2–10x speedup on partitioned SQL loads, 5–50x on string filter operations, and 3–5x on schema coercion pipelines.

---

## Current State Analysis

### Where PyArrow Is Used Today

| Module | Usage | Limitation |
|--------|-------|------------|
| `parquet/resource.py` | `pyarrow.dataset` for file discovery & partitioning | Only extracts file paths; data read goes to Dask/pandas |
| `core/filesystem.py` | `pyarrow.fs` for Local/S3 filesystem bridges | Used as a passthrough, not for compute |
| `data/schema.py` | PyArrow dtype string aliases (`"int64[pyarrow]"`) | Normalization only; no actual Arrow types |
| `core/connection_catalog.py` | Type hints for `pyarrow.fs.FileSystem` | Type annotation only |

### The Core Problem

```
Parquet Path:  Parquet → PyArrow (internal to Dask) → pandas partitions → Dask graph
SQL Path:      SQLAlchemy cursor → pd.DataFrame(rows) → apply_schema_map() → Dask
```

Data **never materializes as a PyArrow Table** in application code. Every operation — filtering, coercion, sorting, aggregation — goes through pandas, which means:
- **Row-by-row pandas construction** from SQLAlchemy cursors (slow for large partitions)
- **Per-column coercion loops** via `pd.to_datetime()`, `pd.to_numeric()`, `.astype()`
- **String operations** via pandas `.str.*` methods (not vectorized)
- **No zero-copy concatenation** for chunked IN-list results

---

## Proposed Integration — Phased Approach

### Phase 1: Arrow-Backed SQL Partition Loading (Highest Impact)

**Target:** `data/db/partitioned_execution.py`, `data/db/sql_model_builder.py`

**Current hot path** (runs for every SQL partition):
```python
# partitioned_execution.py:86-101
frame = pd.DataFrame(rows, columns=columns)
return SqlPartitionExecutor.align_and_coerce_partition(frame, meta_dtypes)
```

**Proposed change:**
```python
import pyarrow as pa

# Build Arrow Arrays directly from column buffers (zero-copy from driver where supported)
arrays = [pa.array([row[i] for row in rows], type=arrow_type) for i, arrow_type in enumerate(arrow_types)]
batch = pa.RecordBatch.from_arrays(arrays, names=columns)
table = pa.Table.from_batches([batch])

# Single-pass coercion instead of per-column loop
table = table.cast(target_arrow_schema)
return table.to_pandas(types_mapper=_get_dtype_mapper)  # or return Arrow directly
```

**Benefits:**
- Eliminates per-row pandas DataFrame construction overhead
- Replaces `apply_schema_map()` column-by-column loop with `table.cast_to_schema()` (single pass)
- Estimated **2–5x speedup** on large partition fetches

**Implementation details:**
1. Add `data/db/arrow_schema_mapper.py` — maps SQLAlchemy types to PyArrow types
2. Modify `SqlPartitionExecutor.fetch_partition()` to build Arrow batches
3. Add configuration flag `use_arrow=True` on `DataGateway` for opt-in migration
4. Maintain backward compatibility — fallback to pandas path when `use_arrow=False`

---

### Phase 2: Arrow Compute Kernels for Filter Operations

**Target:** `data/filters/utils.py`, `data/filters/handler.py`

**Current limitations:**
- String operations (`contains`, `startswith`, `regex`) use pandas `.str.*` via `map_partitions`
- `$or`/`$not` boolean operators excluded from pushdown — applied as residual pandas filters
- Date casting in filters uses `pd.Timestamp` arithmetic

**Proposed changes:**

#### 2a. Residual String Filters via PyArrow Compute

```python
import pyarrow.compute as pc

# Current (slow):
column.map_partitions(lambda s: s.str.contains(pattern, na=False))

# Proposed (fast):
pc.match_substring(column_chunk, pattern=pattern)  # vectorized, zero-copy on Arrow chunk
pc.match_like(column_chunk, pattern)                # for LIKE patterns
pc.match_substring_regex(column_chunk, pattern)     # for regex
```

**Implementation:**
1. Add `data/filters/arrow_kernels.py` — PyArrow compute equivalents for all filter operations
2. Modify `build_mask_fn()` in `handler.py` to detect Arrow-backed columns and use `pc.*` kernels
3. Extend `split_pushdown_and_residual()` to include more operators in pushdown when Arrow is available

#### 2b. $or/$not Boolean Operator Pushdown

```python
# Current: $or returns empty pushdown, all applied as residual
# Proposed: Use PyArrow dataset expressions
import pyarrow.dataset as ds

ds.or_(filter_expr_a, filter_expr_b)
ds.invert(filter_expr)  # for $not
```

**Benefits:**
- **5–50x speedup** on string-heavy residual filters
- Dramatically reduced memory usage (no intermediate pandas partitions for filtered data)
- Better parquet pushdown coverage

---

### Phase 3: Arrow Schema as Canonical Contract

**Target:** `data/schema.py`, `data/field_map.py`

**Current approach:**
```python
# schema.py: dict-based dtype mapping + per-column coercion
SCHEMA_MAP = {"col_a": "Int64", "col_b": "boolean", "col_c": "datetime64[ns, UTC]"}
for col, dtype in schema_map.items():
    df[col] = _coerce_series(df[col], dtype)  # N separate passes
```

**Proposed approach:**
```python
# Define canonical PyArrow schema
ARROW_SCHEMA = pa.schema([
    ("col_a", pa.int64()),
    ("col_b", pa.bool_()),
    ("col_c", pa.timestamp("ns", tz="UTC")),
])

# Single-pass coercion
table = pa.Table.from_pandas(df)
table = table.cast(ARROW_SCHEMA)  # one operation, minimal copy
```

**Implementation:**
1. Add `data/schema/arrow_schema.py` — `ArrowSchema` class wrapping `pa.Schema` with:
   - `from_dict()` constructor from existing dict format
   - `cast_table()` method for single-pass coercion
   - `equals()` for validation (replaces `validate_schema()`)
2. Modify `apply_schema_map()` to use `table.cast_to_schema()` when Arrow is available
3. Add `FieldMap.rename_table()` — `table.rename_columns()` is O(1) (no data movement vs pandas `rename`)

**Benefits:**
- **3–5x speedup** on schema coercion (single pass vs N column loops)
- Stronger schema contracts — `pa.Schema` has built-in equality, nullability, and metadata support
- Cleaner dtype normalization — no more string alias parsing

---

### Phase 4: Arrow-Native DataGateway Path

**Target:** `data/gateway/core.py`, `data/gateway/loaders.py`

**Current:** DataGateway always returns pandas or Dask DataFrames.

**Proposed:** Add `return_type` parameter: `"pandas" | "arrow" | "dask"`

```python
# New signature
gateway.load(
    query="...",
    return_type="arrow",  # new option
    filters={...},
)
# Returns: pa.Table (single partition) or list[pa.Table] (partitioned)
```

**Data flow with Arrow path:**
```
SQL:     SQLAlchemy cursor → Arrow RecordBatch → pa.Table → (optional: to_pandas)
Parquet: pyarrow.dataset → pa.Table → (optional: to_dask / to_pandas)
```

**Implementation:**
1. Add `return_type` parameter to `DataGateway.load()`, `load_all()`, `load_period()`, `semi_join()`, etc.
2. Modify loaders (`data/gateway/loaders.py`) to produce Arrow Tables when requested
3. Add `data/gateway/arrow_adapters.py` — Arrow equivalents of Dask post-load transforms:
   - `sort_by()` via `table.sort_by()`
   - `drop_duplicates()` via `table.group_by().aggregate()`
   - `concat_tables()` via `pa.concat_tables()` (zero-copy)
4. Extend `_apply_df_options()` to `_apply_table_options()` for Arrow path

**Benefits:**
- Users who don't need pandas can skip the conversion entirely
- Downstream Arrow consumers (e.g., DuckDB, Polars, ML pipelines) get native input
- Enables zero-copy interop with other Arrow-compatible libraries

---

### Phase 5: Advanced Optimizations

#### 5a. Parquet Statistics-Aware Partition Planning

**Target:** `data/parquet/resource.py`

Current file discovery only extracts paths. PyArrow datasets expose:
- `dataset.scanner().scan_statistics()` — row counts, null counts, min/max per column
- Fragment-level metadata — file sizes, row group info

**Use cases:**
- Skip empty or irrelevant files before loading
- Better partition sizing based on actual row counts
- Predicate pushdown effectiveness estimation

#### 5b. Arrow Memory-Mapped Reads for Large Parquet Files

```python
import pyarrow.parquet as pq

# Current: full read into memory
table = pq.read_table(path)

# Proposed: memory-map for out-of-core processing
pf = pq.ParquetFile(path, memory_map=True)
table = pf.read_row_group(0)  # only needed row groups
```

#### 5c. Dictionary Encoding for Categorical Columns

```python
# Automatically dictionary-encode low-cardinality columns
table = table.combine_chunks()
table = table.replace_schema_metadata({"pandas_type": "categorical"})
```

Reduces memory footprint 5–20x for columns like `status`, `country`, `type`.

#### 5d. IPC Serialization for Caching

```python
# Arrow IPC format for fast intermediate caching
sink = pa.OSFile(path, "wb")
with pa.ipc.new_file(sink, table.schema) as writer:
    writer.write(table)

# Zero-copy read
with pa.ipc.open_file(path) as reader:
    table = reader.get_all()
```

Faster than parquet for write-once-read-many cached results.

---

## Implementation Roadmap

| Phase | Scope | Estimated Effort | Risk |
|-------|-------|------------------|------|
| **1. Arrow SQL Loading** | `partitioned_execution.py`, new `arrow_schema_mapper.py` | Medium | Low — opt-in, backward compatible |
| **2. Arrow Filter Kernels** | `filters/utils.py`, `filters/handler.py`, new `arrow_kernels.py` | Medium | Low — additive, no breaking changes |
| **3. Arrow Schema Contract** | `schema.py`, `field_map.py`, new `arrow_schema.py` | Low | Low — wraps existing dict format |
| **4. Arrow DataGateway Path** | `gateway/core.py`, `loaders.py`, new `arrow_adapters.py` | High | Medium — new API surface |
| **5. Advanced Optimizations** | Statistics, memory-mapping, dictionary encoding, IPC | Low each | Low — optional features |

**Recommended order:** Phase 1 → Phase 3 → Phase 2 → Phase 4 → Phase 5

Phase 1 gives the highest immediate ROI. Phase 3 strengthens the schema contract with minimal effort. Phase 2 builds on Phase 3's Arrow types. Phase 4 is the largest change but unlocks the full Arrow ecosystem.

---

## Testing Strategy

### New Test Modules
```
tests/data/test_arrow_schema_mapper.py    — SQLAlchemy ↔ Arrow type mapping
tests/data/test_arrow_kernels.py          — PyArrow compute filter equivalence
tests/data/test_arrow_schema_contract.py  — cast_to_schema, validation
tests/data/test_arrow_gateway.py          — return_type="arrow" end-to-end
tests/data/test_arrow_parquet_stats.py    — Statistics-aware partition planning
```

### Regression Testing
- All existing tests pass with `use_arrow=False` (default)
- New tests with `use_arrow=True` verify identical results
- Performance benchmarks comparing pandas vs Arrow paths

### Benchmark Suite
```python
# tests/benchmarks/
test_sql_partition_arrow_vs_pandas()      — Phase 1
test_string_filter_arrow_vs_pandas()      — Phase 2
test_schema_coercion_arrow_vs_pandas()    — Phase 3
test_large_parquet_scan_arrow_vs_pandas() — Phase 4
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **PyArrow version compatibility** | Pin minimum version (`pyarrow>=14.0.0`), test against multiple versions |
| **Memory overhead** | Arrow is columnar — large wide tables may use more memory; use chunked arrays and `combine_chunks()` judiciously |
| **Type mapping gaps** | Start with common types (int, float, string, bool, timestamp); add edge cases incrementally |
| **Breaking changes** | All Arrow features opt-in via `use_arrow=False` default; no existing behavior changes |
| **Dependency bloat** | PyArrow is already a dependency; no new external deps for Phases 1–3 |

---

## Dependencies

**Already present** (in `pyproject.toml`):
- `pyarrow` — core library
- `dask[dataframe]` — uses PyArrow internally for parquet
- `fsspec` — filesystem abstraction

**No new dependencies required** for Phases 1–4.

---

## Conclusion

Boti's architecture is well-positioned for PyArrow integration. The `ManagedResource` lifecycle, `DataGateway` abstraction, and existing PyArrow filesystem usage provide a clean foundation. By introducing Arrow as a first-class in-memory representation — starting with SQL partition loading and schema coercion — Boti can achieve **significant performance gains** without breaking changes or new dependencies.

The phased approach ensures:
1. **Immediate value** (Phase 1: 2–5x SQL load speedup)
2. **Low risk** (opt-in, backward compatible)
3. **Progressive enhancement** (each phase builds on the last)
4. **No new dependencies** (uses existing `pyarrow`)

**Next step:** Implement Phase 1 with a feature flag, benchmark against production workloads, and validate performance gains before proceeding.
