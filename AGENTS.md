# AGENTS.md

## Purpose
- `boti-data` is the data access + transformation layer for SQL and Parquet, with Dask-first scaling.
- Start from `DataHelper` for app-facing usage; `DataGateway` is the core execution engine.

## Architecture Map (read in this order)
- Public API surface: `src/boti_data/__init__.py`
- Facade layer: `src/boti_data/helper.py` (`DataHelper`, engine-bound views: `.dask/.pandas/.polars`)
- Core orchestration: `src/boti_data/gateway/core.py` (return-type resolution, sync/async, configured/structured modes)
- Request/contract models: `src/boti_data/gateway/requests.py`
- SQL partitioned execution: `src/boti_data/db/partitioned_loader.py`, `src/boti_data/db/partitioned_planner.py`, `src/boti_data/db/partitioned_execution.py`
- Parquet backend: `src/boti_data/parquet/resource.py`
- Dask runtime/session/resilience helpers: sibling package `boti-dask` (`src/boti_dask/session.py`, `src/boti_dask/resilience.py`, `src/boti_dask/diagnostics.py`)
- Join primitives: `src/boti_data/joins.py`

## Project-Specific Conventions
- `field_map` direction is **DB -> semantic** (not semantic -> DB): see `src/boti_data/field_map.py`.
- Filters and configured `fieldnames` are expressed with semantic names; translation to DB names is internal.
- `DataGateway` has two modes:
  - structured mode: pass `statement` + `model` + optional `filters`
  - configured mode: set `table=` at construction; non-control kwargs become runtime filters
- Control kwargs are centrally defined in `LOAD_CONTROL_KEYS`: `src/boti_data/gateway/normalization.py`.
- `return_type="auto"` is size-aware and can probe SQL/parquet to choose pandas vs dask (threshold logic in `gateway/core.py`).
- `helper.pandas`, `helper.polars`, and `helper.dask` enforce compatible `return_type`/`execution_mode` and raise on conflicts.

## Integration Boundaries
- SQL config and worker-safe serialization live in `src/boti_data/db/sql_config.py` (`SqlDatabaseConfig`, `WorkerSqlConfig`).
- For distributed SQL, prefer `worker_connection_env_var` so DSNs are resolved on workers (avoid pickling credentials).
- Parquet distributed access reconstructs filesystem from `fs_factory`/`filesystem_profile`; use `ConnectionCatalog` for named profiles.
- `query_only` materially affects engine identity and security expectations (see `tests/security/test_regressions.py`).
- `boti_data` consumes runtime primitives from `boti_dask`; avoid adding new session/resilience internals back into `src/boti_data`.

## Developer Workflows
- Install dev deps (from `Makefile`):
```bash
uv pip install -e ".[dev]"
```
- Run tests:
```bash
uv run pytest
uv run pytest tests/data/test_facade.py -k auto_return_type
uv run pytest -m security_regression
```
- Run examples from repo root (see `examples/README.md`):
```bash
python examples/data_facade_db.py
python examples/data_facade_distributed.py
python examples/data_helper_distributed.py
```

## High-Value Debug Patterns
- Turn on `diagnostics=True` in gateway loads and joins to log plan/partition/client summaries.
- For large Dask joins, prefer `indexed_left_join(...)` over direct merge; align join-key dtypes via `join_schema_map`.
- If lazy SQL load fails, verify you supplied `statement` + `model` (raw `sql=` is eager-oriented).

## AI Coding Agent Guidance
- When adding new features, ensure backward compatibility with existing `DataHelper` and `DataGateway` interfaces.
- Prefer composition over inheritance when extending functionality; utilize existing components in `src/boti_data` as building blocks.
- Adhere to the established directory structure and naming conventions for new modules or functions.
- Include comprehensive docstrings and type annotations for all new public methods or classes.
- For performance-sensitive code, consider the impact of Dask's lazy evaluation and parallel execution model.
- Leverage the existing test suite as a guide for expected behavior and edge cases; aim for high test coverage on new code.
- Consult the architecture map and project-specific conventions regularly to ensure alignment with the overall system design.
