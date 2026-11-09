# Security Hardening Guide

This guide documents defensive defaults and validation hooks for SQL and Datacube paths in `boti-data`.

## Scope

- SQL reads through `DataGateway` and `DataHelper`
- Datacube request validation via `DatacubeContract`
- Distributed SQL worker credential handling

## Filter Interpolation Hotspots

Use this matrix during reviews to identify where dynamic filter/query inputs are accepted and what control should be enforced.

| Module | Entry point | Risk type | Recommended control |
| --- | --- | --- | --- |
| `src/boti_data/gateway/normalization.py` | `split_control_and_filters()`, `normalize_configured_filters()` | Runtime kwargs not in `LOAD_CONTROL_KEYS` are treated as filters and merged with sticky/explicit filters. | Restrict caller-provided kwargs shape and validate allowed filter keys at API boundary. |
| `src/boti_data/gateway/core.py` | `_build_configured_request()` | Merged semantic filters are forwarded into SQL query construction. | Enforce semantic field allowlists and keep `field_map` translation enabled where DB names differ. |
| `src/boti_data/field_map.py` | `translate_filters_to_db()` | Dynamic filter keys are rewritten into DB-column keys, including nested boolean operators. | Validate semantic keys before translation; reject unknown/suspicious operator forms. |
| `src/boti_data/filters/handler.py` | `compile_filters()`, `apply_filters()` | Dynamic `field__op` expressions compile into backend-specific conditions. | Keep operator parsing strict and test forbidden keys/ops in security regressions. |
| `src/boti_data/gateway/loaders.py` | `_prepare_sql_statement()` | Raw `sql` text is accepted when allowed; app-side string interpolation can bypass intent. | Prefer `raw_sql_policy="disabled"` for production and require bound `params` for any approved raw SQL. |
| `src/boti_data/parquet/reader.py` | `_normalize_filter_options()` | Bare kwargs are promoted into `filters`, which may broaden query scope unexpectedly. | Use explicit `filters=` in higher-risk call paths and lint/review wildcard runtime kwargs. |
| `src/boti_data/datacube/resource.py` | `_prepare_request()` | Datacube filters/cube names are accepted unless contract validator is configured. | Make `DatacubeContract.request_validator` mandatory in production configs. |

## SQL Hardening Defaults

### Raw SQL Is Denied by Default

`sql=` execution requires an explicit `allow_raw_sql=True` opt-in. Without that flag, loads fail fast.

### Raw SQL Is Read-Only Only

Even when enabled, only single-statement read-only `SELECT`/`WITH` queries are accepted.
Mutating or multi-statement payloads are rejected.

### Disable Raw SQL Entirely

For stricter deployments, initialize `DataGateway(..., raw_sql_policy="disabled")`.
This blocks all `sql=` calls, including those with `allow_raw_sql=True`.

### Always Bind Parameters

Use bound parameters instead of string interpolation:

```python
frame = gateway.load(
    sql="SELECT id, status FROM users WHERE status = :status",
    params={"status": status_value},
    as_pandas=True,
    allow_raw_sql=True,
)
```

Do **not** build SQL with f-strings or concatenated user input.

## Datacube Hardening

Datacube security policy is contract-driven.
If no `request_validator` is configured, suspicious cubes/filter keys can be accepted by permissive loaders.

Use a strict validator to enforce business and security constraints:

```python
def request_validator(request):
    cube_name = str(request.cube or "")
    if ".." in cube_name or "/" in cube_name:
        raise ValueError("cube name contains forbidden path tokens")
    if any(str(key).startswith("$") for key in request.filters):
        raise ValueError("operator-style filter keys are not allowed")
```

Attach it with `DatacubeContract(request_validator=...)`.

You can enforce validator presence at gateway construction time:

```python
gateway = DataGateway(
    DatacubeConfig(loader=loader, contract=contract),
    require_datacube_request_validator=True,
)
```

## Strict Filter Validation (Opt-In)

`DataGateway` supports optional strict controls for configured-mode and datacube filter payloads:

- `strict_filter_validation=True`
- `allowed_filter_fields={...}`
- `max_filter_depth=...`
- `max_filter_conditions=...`
- `max_in_filter_values=...`

Example:

```python
gateway = DataGateway(
    sql_config,
    table="users",
    strict_filter_validation=True,
    allowed_filter_fields={"status", "created_at"},
    max_filter_depth=6,
    max_filter_conditions=64,
    max_in_filter_values=1000,
)
```

## Distributed SQL Credential Handling

For distributed execution, set `worker_connection_env_var` on `SqlDatabaseConfig` so task payloads do not serialize raw DSNs:

```python
config = SqlDatabaseConfig(
    connection_url="postgresql+psycopg://...",
    worker_connection_env_var="DB_DSN",
)
```

Ensure the configured env var is present on every worker.

## Regression Coverage

Security-focused regressions live in:

- `tests/security/test_regressions.py`

Run only security regressions with:

```bash
uv run pytest -m security_regression
```

