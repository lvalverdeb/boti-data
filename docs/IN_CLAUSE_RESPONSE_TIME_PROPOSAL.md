# IN-Clause Response Time Proposal (`boti_data.db`)

## Objective

Improve query response times when `WHERE ... IN (...)` filters are used, starting from a measured baseline of current behavior.

## Code Paths Reviewed

Primary paths involved in `IN` handling and query execution:

- `src/boti_data/filters/utils.py`
  - `apply_in_sqlalchemy(...)`
  - `normalize_in_filter_values(...)`
  - `suggest_in_filter_chunking(...)`
- `src/boti_data/filters/handler.py`
  - `FilterHandler.suggest_sql_in_chunking(...)`
- `src/boti_data/gateway/core.py`
  - `_resolve_in_chunk_controls(...)`
  - `_chunked_in_load_sync(...)`
  - `_chunked_in_load(...)`

## Baseline Benchmark

### Harness

A reproducible benchmark script was added:

- `benchmarks/benchmark_in_clause_baseline.py`

Run it with:

```bash
uv run python benchmarks/benchmark_in_clause_baseline.py
```

### Benchmark Conditions

- Backend: SQLite (local file DB)
- Rows: `200,000`
- `IN` cardinalities: `100`, `1,000`, `5,000`, `10,000`
- Repeats: `3` (median reported)
- Scenarios:
  - sync pandas, single query (`in_chunk_strategy="off"`)
  - sync pandas, auto chunking
  - sync pandas, forced chunking
  - sync dask, single query
  - sync dask, auto chunking
  - async pandas, single query
  - async pandas, auto chunking

## Current Response Times (median ms)

| Scenario | IN=100 | IN=1,000 | IN=5,000 | IN=10,000 |
|---|---:|---:|---:|---:|
| `sync_pandas_single` | 1.77 | 3.94 | 14.34 | 26.36 |
| `sync_pandas_auto_chunk` | 1.53 | 5.22 | 57.50 | 186.69 |
| `sync_pandas_forced_chunk` | 1.14 | 4.02 | 32.71 | 73.38 |
| `sync_dask_single` | 3.46 | 11.00 | 125.30 | 444.57 |
| `sync_dask_auto_chunk` | 3.85 | 19.74 | 86.07 | 173.06 |
| `async_pandas_single` | 1.40 | 3.99 | 14.66 | 32.34 |
| `async_pandas_auto_chunk` | 2.09 | 5.12 | 58.60 | 185.65 |

## Phase 1 Implementation Status

Implemented in `src/boti_data/gateway/core.py`:

- Auto chunking now receives resolved execution mode (`eager`/`lazy`).
- Eager SQL auto chunking is disabled for medium IN lists on SQLite.
- Eager SQL auto chunking threshold is now dialect-aware:
  - SQLite: auto chunk only when `IN >= 8,000`.
  - Other SQL backends: fallback threshold `5,000`.
- Eager SQL chunk fan-out concurrency is capped to `1` when auto chunking does apply.

Unit tests added:

- `tests/data/test_facade.py::test_facade_auto_in_chunking_disables_eager_sql_for_medium_in_lists`
- `tests/data/test_facade.py::test_facade_auto_in_chunking_keeps_lazy_sql_chunking_for_medium_in_lists`

## Post-Phase-1 Benchmark Snapshot (median ms)

| Scenario | IN=100 | IN=1,000 | IN=5,000 | IN=10,000 |
|---|---:|---:|---:|---:|
| `sync_pandas_single` | 1.24 | 3.45 | 14.07 | 26.16 |
| `sync_pandas_auto_chunk` | 1.67 | 4.24 | 18.48 | 42.50 |
| `async_pandas_single` | 1.38 | 4.13 | 15.20 | 27.74 |
| `async_pandas_auto_chunk` | 2.12 | 4.78 | 15.86 | 42.18 |
| `sync_dask_single` | 3.66 | 12.26 | 128.29 | 437.40 |
| `sync_dask_auto_chunk` | 3.77 | 20.50 | 85.51 | 173.07 |

### Observed impact vs baseline

- Eager auto chunking improved significantly for large IN lists:
  - `sync_pandas_auto_chunk`, IN=10k: **186.69 ms -> 42.50 ms** (~4.4x faster).
  - `async_pandas_auto_chunk`, IN=10k: **185.65 ms -> 42.18 ms** (~4.4x faster).
- Lazy Dask auto chunking remained beneficial for larger IN lists.
- Acceptance target (`<=1.5x` of single-query eager baseline) is now close but not met at IN=10k:
  - `42.50 / 26.16 = 1.62x`.

## Phase 2 Implementation Status

Implemented in `src/boti_data/gateway/core.py`:

- Added adaptive policy object for eager auto chunking decisions (`_InChunkPolicy`).
- Policy now resolves by SQL dialect (SQLite gets a stricter eager threshold).
- Added chunk fan-out diagnostics telemetry for sync and async paths:
  - chunk count
  - total fan-out elapsed
  - per-chunk elapsed list
  - log labels: `Gateway IN chunk fan-out sync|async`

Additional tests added:

- `tests/data/test_facade.py::test_facade_auto_in_chunking_disables_eager_sql_for_ten_thousand_sqlite_values`
- `tests/data/test_facade.py::test_facade_in_chunk_diagnostics_logs_sync_fanout`

## Post-Phase-2 Benchmark Snapshot (median ms)

| Scenario | IN=100 | IN=1,000 | IN=5,000 | IN=10,000 |
|---|---:|---:|---:|---:|
| `sync_pandas_single` | 1.35 | 3.51 | 14.12 | 27.36 |
| `sync_pandas_auto_chunk` | 1.59 | 4.15 | 18.03 | 28.62 |
| `async_pandas_single` | 1.46 | 4.06 | 15.26 | 40.03 |
| `async_pandas_auto_chunk` | 1.99 | 4.62 | 16.83 | 33.07 |
| `sync_dask_single` | 3.68 | 11.22 | 127.32 | 446.81 |
| `sync_dask_auto_chunk` | 4.21 | 19.82 | 85.58 | 180.54 |

### Phase-2 impact summary

- `sync_pandas_auto_chunk` at IN=10k now lands near single-query latency:
  - **28.62 ms vs 27.36 ms** (`1.05x`, acceptance achieved).
- `sync_pandas_auto_chunk` at IN=5k remains within target:
  - **18.03 ms vs 14.12 ms** (`1.28x`, acceptance achieved).
- Lazy Dask auto chunking remains beneficial at larger IN sizes.

## Findings

1. For eager/pandas paths, chunking is a major regression once `IN` grows:
   - `IN=10,000`: `sync_pandas_auto_chunk` is ~7x slower than single-query.
   - Similar degradation on async eager (`async_pandas_auto_chunk`).
2. For lazy/dask paths, chunking helps at larger cardinalities:
   - `IN=10,000`: `sync_dask_auto_chunk` is ~2.6x faster than single-query.
3. Current auto threshold (`~900`) is too low for eager paths on this backend.
4. The same chunk policy is being applied without enough sensitivity to:
   - execution mode (`eager` vs `lazy`)
   - backend/dialect characteristics
   - crossover point where chunk fan-out starts paying off.

## Proposal

### Phase 1: Low-Risk Tuning (recommended first)

1. Make auto chunking execution-mode aware in `DataGateway`:
   - For eager paths (`pandas`/`polars`):
     - default to `in_chunk_strategy="off"` unless `IN` is very large.
   - For lazy/dask paths:
     - keep chunking enabled.
2. Raise eager auto chunk activation threshold (dialect-specific):
   - SQLite baseline suggestion: do not auto-chunk until at least `IN >= 8,000`.
3. Cap eager chunk concurrency at `1` by default:
   - Avoid thread fan-out overhead for local DB workloads.

Expected impact:

- Substantial latency reduction for large eager `IN` filters (4x-7x in baseline patterns).

### Phase 2: Adaptive Policy

1. Introduce a backend/dialect policy object, e.g.:
   - `sqlite`: high eager threshold, lower concurrency
   - `postgres/mysql`: threshold tuned from backend benchmarks
2. Add crossover telemetry in diagnostics logs:
   - chunk count
   - per-chunk elapsed
   - total fan-out overhead

Expected impact:

- Better default behavior across environments without manual tuning.

### Phase 3: Very Large IN Strategy

For very large lists (e.g., >20k to 50k, backend-dependent), evaluate:

1. temporary table + join
2. backend-native array strategies (where supported)

Expected impact:

- Improved plan stability and lower SQL/parameter overhead for extreme cardinalities.

## Rollout Plan

1. Implement Phase 1 tuning behind safe defaults.
2. Re-run benchmark matrix and compare to this baseline.
3. Add regression benchmarks to CI-adjacent perf checks (non-blocking initially).
4. Progress to Phase 2 after validating no regressions in distributed/lazy paths.

## Acceptance Criteria

For the same baseline harness:

- `sync_pandas_auto_chunk` at `IN=5,000` and `IN=10,000` should be within `1.5x` of `sync_pandas_single` (or better).
- `sync_dask_auto_chunk` should not regress at `IN=5,000` and `IN=10,000`.
- No correctness deltas in returned row counts.

## Notes

- These numbers are from a local SQLite benchmark and are intended as a baseline.
- Final tuning should be validated against your production DB engine and network profile.

