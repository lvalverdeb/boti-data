# Dask resilience and cluster-lifecycle use cases for `boti-data`

## Summary

Reviewing `sibi_flux.dask_cluster` and `sibi_dst.utils.dask_resilience` shows a consistent theme: both packages add a **resilience layer above ordinary Dask usage** rather than replacing Dask itself. The main ideas are:

- keep a healthy client available
- retry on recoverable comm failures
- avoid unsafe fallback for Dask DataFrame graphs
- expose graph and cluster diagnostics
- optionally share or heal cluster state across processes

`boti-data` now relies on `boti_dask` for this runtime foundation:

- `../boti-dask/src/boti_dask/session.py` for explicit client/session ownership
- `../boti-dask/src/boti_dask/resilience.py` for resilient wrappers and emptiness helpers
- `../boti-dask/src/boti_dask/diagnostics.py` for graph/frame diagnostics
- `src/boti_data/gateway/core.py` for load diagnostics and client summaries
- `src/boti_data/joins.py` for diagnostics around large distributed joins
- `src/boti_data/db/partitioned_loader.py` for worker-safe partitioned SQL loading

That means the best opportunity is **selective adoption**: add resilience and observability around high-value distributed operations without importing the full lifecycle model from the reviewed projects.

## Current implementation status

The prioritized opt-in resilience slice is now implemented via `boti_dask` and consumed by `boti-data`.

Available today:

- `inspect_graph(...)`
- `safe_compute(...)`
- `safe_head(...)`
- `safe_gather(...)`
- `safe_persist(...)`
- `safe_wait(...)`
- `async_safe_compute(...)`
- `async_safe_head(...)`
- `async_safe_gather(...)`
- `async_safe_persist(...)`
- `async_safe_wait(...)`
- `dask_is_probably_empty(...)`
- `dask_is_empty(...)`
- `UniqueValuesExtractor`
- `recommended_dask_config(...)`
- `apply_recommended_dask_config(...)`
- `DataGateway.load(..., dry_run=True)` / `aload(..., dry_run=True)` for lazy Dask preflight
- `DataGateway.preview(...)` / `apreview(...)`
- `DataHelper.preview(...)` / `apreview(...)`
- `indexed_left_join(..., dry_run=True)` and `left_join_frames(..., dry_run=True)`
- opt-in Dask session connectivity verification
- opt-in shared session reuse
- `indexed_left_join(..., resilient=True)`
- gateway diagnostics that include Dask graph metrics

Still pending or intentionally deferred:

- registry/watchdog lifecycle management
- worker code deployment

## What the reviewed modules contribute

## `sibi_flux.dask_cluster`

Most relevant patterns:

- **persistent/shared client registry** via a temp-file registry and refcounting
- **connectivity verification** before accepting a scheduler/client as healthy
- **watchdog-based healing** when scheduler comms fail
- **safe compute/persist/gather wrappers** with one retry after client rebind
- **strict invariant**: never silently fall back to local threaded compute for Dask DataFrame/Series graphs
- **graph inspection / dry run** to log task count, layers, and partition counts before execution
- **optional local code deployment** to workers for remote scheduler scenarios

This module is the stronger reference for **cluster lifecycle and operational safety**.

## `sibi_dst.utils.dask_resilience`

Most relevant patterns:

- **global Dask config hardening**, especially `dataframe.shuffle.method = "tasks"`
- **narrower recoverable exception policy**
- **sync and async safe wrappers** for compute/persist/gather/wait
- **cheap emptiness probes** for Dask collections
- **resilient unique value extraction**

This module is the stronger reference for **execution wrappers and defensive data utilities**.

## Current `boti-data` baseline

`boti-data` already supports (directly or through `boti_dask`):

- explicit `DaskSession` / `dask_session(...)`
- local or remote scheduler attachment
- diagnostics for gateway loads and joins
- partitioned SQL execution designed for worker-local engines
- Dask-first joins through `indexed_left_join(...)`

What it does **not** currently provide:

- automatic client healing or rebind-and-retry
- resilient wrappers around expensive distributed actions
- persistent shared-cluster reuse across notebook/process boundaries
- explicit dry-run graph inspection before large actions
- Dask-specific helpers like resilient emptiness/unique-value probes

## High-value use cases for `boti-data`

## 1. Resilient materialization of expensive Dask loads

### Problem

Large `DataGateway.load(..., return_type="dask")` workloads often end in a small number of materialization points:

- `.persist()` after repartition or indexing
- `.compute()` for exports or model inputs
- `.head()` / sample extraction for notebook inspection

If the client drops during these steps, users currently have to manage recovery themselves.

### Use case

Add opt-in helpers such as:

- `safe_compute(frame, client=None, logger=None)`
- `safe_persist(frame, client=None, logger=None)`
- `safe_wait(obj, client=None, timeout=None)`

These should:

1. detect an active client
2. retry once on recoverable comm failures
3. refuse unsafe local fallback for Dask DataFrames
4. surface a clear error instructing the caller to rebuild the collection when the graph is orphaned

### Likely touchpoints

- `../boti-dask/src/boti_dask/resilience.py`
- `src/boti_data/gateway/core.py`
- `src/boti_data/joins.py`

### Value

High. This directly protects the most expensive user-visible operations in `boti-data`.

## 2. Safer post-join and post-index workflows

### Problem

`indexed_left_join(...)` already recommends `persist=True` for expensive join pipelines, but the actual persist step is still a plain Dask action. If a cluster hiccup happens after `set_index(...)`, the user may face recomputation or ambiguous failures.

### Use case

Wrap the expensive boundary operations used by distributed joins:

- optional resilient persist after `set_index(...)`
- resilient persist on the final joined frame
- clearer diagnostics when a join graph becomes invalid after cluster recreation

### Likely touchpoints

- `src/boti_data/joins.py`

### Value

High. Join-heavy workflows are one of the clearest “big data” use cases in this repo.

## 3. Dry-run graph inspection for large distributed operations

### Problem

`boti-data` already logs frame summaries after work is planned or completed, but it does not provide a first-class way to inspect graph size before execution.

### Use case

Add an optional dry-run / inspection helper for Dask-backed frames that logs:

- task count
- number of partitions
- number of graph layers
- active client summary

This could be used in:

- `DataGateway.load(..., diagnostics=True, dry_run=True)` for distributed loads
- `indexed_left_join(..., diagnostics=True, dry_run=True)` before a large shuffle/index stage

### Likely touchpoints

- `../boti-dask/src/boti_dask/diagnostics.py`
- `src/boti_data/gateway/core.py`
- `src/boti_data/joins.py`

### Value

Medium-high. This would improve notebook ergonomics and preflight checks for production jobs.

## 4. Shared cluster/session reuse in notebook and ETL workflows

### Problem

`boti-data` has explicit session ownership, which is clean and predictable, but repeated notebook cells or multi-step ETL scripts may repeatedly reconnect or recreate clients.

### Use case

Introduce an **optional** shared-session mode on top of `DaskSession`, for example:

- reuse the current context client if one exists
- otherwise attach to a configured scheduler address
- optionally keep a process-local singleton client for repeated operations

This does **not** require the full file-registry model from `sibi_flux` on day one.

### Likely touchpoints

- `../boti-dask/src/boti_dask/session.py`
- `src/boti_data/helper.py`

### Value

Medium. Good fit for notebook-heavy and service-like workloads, but should remain opt-in.

## 5. Defensive Dask defaults for shuffle-heavy pipelines

### Problem

The reviewed modules explicitly harden Dask config for comm timeouts, memory spill thresholds, and shuffle behavior. `boti-data` currently leaves these concerns largely to caller environment and cluster configuration.

### Use case

Provide a documented or opt-in config profile for large distributed `boti-data` workloads, especially:

- `dataframe.shuffle.method = "tasks"`
- conservative comm timeout defaults
- memory spill / pause thresholds
- large-graph warning thresholds

This is especially relevant for:

- `indexed_left_join(...)`
- partitioned SQL loads feeding downstream joins
- large parquet scans followed by repartition or groupby

### Likely touchpoints

- `../boti-dask/src/boti_dask/resilience.py`
- docs and example notebooks

### Value

Medium-high. Likely the simplest way to reduce operational failures without invasive code changes.

## 6. Resilient emptiness and sampling helpers

### Problem

Big Dask frames are often probed for:

- “is this empty?”
- “give me representative unique values”
- “can I cheaply inspect a few results before computing all of it?”

Naive implementations can be either too expensive or fragile.

### Use case

Add small utilities such as:

- `dask_is_probably_empty(frame)`
- `dask_is_empty(frame, sample=4, client=None)`
- `extract_unique_values(frame, *columns, limit=...)`

These would be useful in:

- UI/data validation layers built on top of `DataHelper`
- diagnostics notebooks
- data quality checks before exporting or joining

### Likely touchpoints

- `../boti-dask/src/boti_dask/resilience.py`

### Value

Medium. Useful, but secondary to resilient compute/persist/wait.

## 7. Remote-cluster bootstrap and worker code deployment

### Problem

`sibi_flux` includes automatic package zip/upload to workers and registry-driven cluster healing. That is useful when workers do not automatically have the same Python package available.

### Use case in `boti-data`

Limited. `boti-data` is primarily a library, and worker environment packaging is usually better solved through deployment tooling, container images, or environment management rather than runtime zip upload.

### Value

Low for near-term adoption. This is probably **not** the first place to invest.

## Recommended adoption order

| Priority | Candidate | Why it fits `boti-data` |
|---|---|---|
| 1 | resilient `compute/persist/wait` wrappers | protects the most expensive distributed actions with minimal API surface |
| 2 | join-boundary resilience | directly improves `indexed_left_join(...)` workflows |
| 3 | dry-run graph inspection | complements existing diagnostics cleanly |
| 4 | opt-in Dask config profile | low-cost operational improvement |
| 5 | emptiness/unique-value helpers | useful utilities for notebook and validation workflows |
| 6 | shared singleton session mode | helpful, but should stay optional |
| 7 | registry/watchdog + code deployment | operationally heavier than current repo needs |

## Design guidance for `boti-data`

## Prefer wrappers over automatic magic

The reviewed packages are comfortable with hidden healing and shared global state. `boti-data` currently favors more explicit control. The best fit is:

- explicit session ownership by default
- opt-in resilience helpers
- strong diagnostics when healing or retry happens
- no silent fallback from distributed DataFrame graphs to local execution

## Keep DataFrame safety as an invariant

One of the best ideas in both reviewed modules is the rule:

> if a Dask DataFrame graph loses its client or worker dependencies, do not pretend local compute is a safe fallback.

That rule aligns well with `boti-data`’s big-data positioning and should be preserved if resilience helpers are added.

## Reuse existing diagnostics surface

`boti-data` already has:

- `current_client_summary()`
- `describe_frame(...)`
- gateway load diagnostics
- join diagnostics

New resilience features should plug into that existing logging model instead of creating a parallel system.

## Concrete near-term proposal

The first increment was implemented in `boti_dask`, which now provides:

1. `safe_compute`
2. `safe_persist`
3. `safe_wait`
4. `inspect_graph`
5. an opt-in Dask config helper for large shuffles

Then wire those helpers into:

- `indexed_left_join(..., persist=True, diagnostics=True)`
- large `DataGateway` distributed loads when `diagnostics=True`
- examples and docs for notebook / ETL usage

This would capture most of the practical value from the reviewed modules without importing the heaviest parts of their lifecycle model.

## Final assessment

The reviewed modules are relevant to `boti-data`, but **not as a direct transplant**.

The strongest `boti-data` use cases are:

- resilient compute/persist/wait around large Dask frames
- safer distributed join workflows
- graph inspection before expensive execution
- documented Dask hardening defaults for shuffle-heavy analytics

The weakest use cases are:

- full registry-based cluster lifecycle management
- automatic worker code deployment

In short, `boti-data` should borrow the **execution-safety and observability ideas** first, while keeping cluster ownership explicit and lightweight.
