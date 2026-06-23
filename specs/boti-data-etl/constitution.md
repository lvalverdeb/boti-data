# boti-data ETL Framework Constitution

## Core Principles

### I. Dask-First Lazy Evaluation

Data loading defaults to Dask DataFrames for lazy, distributed execution. Eager (pandas) returns are opt-in via explicit `return_type="pandas"` or size-probing auto-detection. This ensures the framework scales from single-machine prototypes to cluster deployments without code changes.

### II. Read/Write Separation

`DataGateway` handles only loading (read). `SinkPipeline` / `ParquetSink` handle materialization (write). No single component both reads and writes data. This keeps the gateway focused on query planning and backend dispatch.

### III. Pushdown Filtering

Filters are split into pushdown (applied at the storage layer — SQL WHERE, Arrow dataset filter) and residual (applied in-memory after load). Every backend must expose a filter split path; no filter should be applied eagerly if the backend can handle it natively.

### IV. Async-First Design

All major operations have sync, async, and `*_sync()` convenience wrappers. The async path is canonical; sync wrappers call through to async via `asyncio.run()` or loop bridging. This enables use in both synchronous scripts and async web frameworks.

### V. Two Gateway Modes

- **Structured mode**: pass `statement` + `model` explicitly with optional `filters`.
- **Configured mode**: set `table=` at construction; runtime non-control kwargs become filters.

Both modes must produce the same internal `LoadRequest` representation.

### VI. Test-First (NON-NEGOTIABLE)

Tests must be written and FAIL before implementation begins. Red-Green-Refactor cycle strictly enforced:
1. Write test for expected behavior
2. Verify test fails
3. Implement
4. Verify test passes

### VII. Simplicity & YAGNI

Start simple. No repository pattern unless cross-backend abstraction is proven necessary. No ORM mapping layer beyond what SQLAlchemy model builder provides. No caching layer until performance profiling demonstrates a bottleneck.

## Backend Contracts

### SQL Backend (`SqlDatabaseResource`)

- All connections read-only by default (`query_only=True`)
- DSN stored as `pydantic.SecretStr` — never logged or serialized unsafely
- Worker-safe config via `WorkerSqlConfig` with `worker_connection_env_var`
- Partitioned loading splits on a partition column (id range or date range)

### Parquet Backend (`ParquetDataResource`)

- Hive partitioning support: `partition_date=2026-01-01/file.parquet`
- File discovery through `pyarrow.dataset` with fallback to manual glob
- Path traversal protection via `SecureResource` with `allowed_paths` validation

### Datacube Backend (`DatacubeResource`)

- Contract-based: `DatacubeConfig` wraps a `DatacubeContract`
- Least-used backend; primarily for pre-aggregated OLAP data

## Quality Gates

### Gate G1 — Constitution Check

Before any planning begins, verify the feature:
- Does not violate any core principle
- Justifies any complexity beyond current patterns
- Has a clear "why" that cannot be met by existing components

### Gate G2 — Specification Completeness

Before planning can proceed, the spec must have:
- All user stories prioritized (P1/P2/P3)
- Acceptance scenarios in Given-When-Then format
- Functional requirements numbered (FR-001, FR-002...)
- Success criteria measurable and technology-agnostic

### Gate G3 — Test Gate

Before implementation begins:
- All test files exist and compile/parse
- All tests fail as expected (red)
- No implementation code exists for the new feature

### Gate G4 — Review Gate

Before merge:
- All tests pass (green)
- Constitution compliance verified
- No credentials, secrets, or paths committed
- No unused dependencies or dead code

## Governance

The constitution supersedes all other practices and technical preferences. Amendments require:
1. Documented rationale in the spec
2. Approval by project maintainer
3. Migration plan for existing code
4. Version bump and changelog entry

All PRs and reviews must verify constitution compliance. Complexity must be justified in the plan's Complexity Tracking section.

**Version**: 1.0.0 | **Ratified**: 2026-06-23 | **Last Amended**: 2026-06-23
