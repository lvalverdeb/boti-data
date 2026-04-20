"""
Security regression tests for recently fixed audit findings.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from boti_data import DatacubeConfig, DatacubeContract
from boti_data.db.sql_config import SqlDatabaseConfig, WorkerSqlConfig
from boti_data.db.sql_manager import AsyncSqlDatabaseResource, EngineRegistry, SqlDatabaseResource
from boti_data.field_map import FieldMap
from boti_data.filters.utils import validate_regex_pattern
from boti_data.gateway import DataGateway

pytestmark = pytest.mark.security_regression


def test_sync_engine_registry_separates_query_only_from_writable_configs():
    """Verify a writable engine is never reused for a query_only resource, or vice versa."""
    query_only_config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:pass@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
    )
    writable_config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:pass@localhost/test_db",
        query_only=False,
        poolclass="sqlalchemy.pool.NullPool",
    )

    readonly_resource = SqlDatabaseResource(query_only_config)
    writable_resource = SqlDatabaseResource(writable_config)

    try:
        assert readonly_resource._engine is not writable_resource._engine
        assert readonly_resource._engine_key != writable_resource._engine_key
    finally:
        readonly_resource.close()
        writable_resource.close()


@pytest.mark.asyncio
async def test_async_engine_registry_separates_query_only_from_writable_configs():
    """Verify async engine caching never mixes query_only and writable resources."""
    query_only_config = SqlDatabaseConfig(
        connection_url="mysql+asyncmy://user:pass@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
    )
    writable_config = SqlDatabaseConfig(
        connection_url="mysql+asyncmy://user:pass@localhost/test_db",
        query_only=False,
        poolclass="sqlalchemy.pool.NullPool",
    )

    async with AsyncSqlDatabaseResource(query_only_config) as readonly_resource:
        readonly_key = readonly_resource._engine_key
        readonly_engine = readonly_resource._engine

        async with AsyncSqlDatabaseResource(writable_config) as writable_resource:
            assert readonly_engine is not writable_resource._engine
            assert readonly_key != writable_resource._engine_key


def test_query_only_and_writable_registry_entries_can_coexist_without_collision():
    """Verify registry bookkeeping keeps readonly and writable entries isolated."""
    query_only_config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:pass@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
    )
    writable_config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:pass@localhost/test_db",
        query_only=False,
        poolclass="sqlalchemy.pool.NullPool",
    )

    readonly_resource = SqlDatabaseResource(query_only_config)
    writable_resource = SqlDatabaseResource(writable_config)

    try:
        assert readonly_resource._engine_key in EngineRegistry._registry
        assert writable_resource._engine_key in EngineRegistry._registry
        assert readonly_resource._engine_key != writable_resource._engine_key
    finally:
        readonly_resource.close()
        writable_resource.close()


# ---------------------------------------------------------------------------
# Worker credential serialisation tests
# ---------------------------------------------------------------------------


def test_worker_config_uses_env_var_when_set():
    """WorkerSqlConfig must NOT carry the raw DSN when worker_connection_env_var is configured."""
    config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:secret@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
        worker_connection_env_var="DB_DSN",
    )
    worker = WorkerSqlConfig.from_database_config(config)
    assert worker.connection_env_var == "DB_DSN"
    assert worker.connection_url is None, "DSN must not be serialized when env var is set"


def test_worker_config_warns_when_no_env_var_set():
    """WorkerSqlConfig.from_database_config must emit a UserWarning when falling back to raw DSN."""
    config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:secret@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        worker = WorkerSqlConfig.from_database_config(config)

    credential_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "worker_connection_env_var" in str(w.message)
    ]
    assert credential_warnings, "Expected a UserWarning about worker_connection_env_var not being set"
    assert worker.connection_url is not None  # fallback still works


def test_worker_config_raw_dsn_fallback_carries_credentials():
    """Sanity-check: when no env var is configured the raw DSN is present (so callers are aware)."""
    config = SqlDatabaseConfig(
        connection_url="mysql+pymysql://user:secret@localhost/test_db",
        query_only=True,
        poolclass="sqlalchemy.pool.NullPool",
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        worker = WorkerSqlConfig.from_database_config(config)

    assert worker.connection_env_var is None
    secret_val = worker.connection_url.get_secret_value()  # type: ignore[union-attr]
    assert "secret" in secret_val


@pytest.mark.parametrize(
    "payload",
    [
        "active'; DROP TABLE users; --",
        "active' OR 1=1 --",
        "active' UNION SELECT 1, 'x' --",
        "active' /* inline */ OR '1'='1",
    ],
)
def test_raw_sql_params_do_not_allow_statement_injection(tmp_path, payload):
    db_path = tmp_path / "sql_injection_guard.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, status TEXT)"))
            conn.execute(text("INSERT INTO users (status) VALUES ('active')"))
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        frame = facade.load(
            sql="SELECT id, status FROM users WHERE status = :status",
            params={"status": payload},
            as_pandas=True,
            allow_raw_sql=True,
        )

    assert frame.empty

    verify_engine = create_engine(f"sqlite:///{db_path}")
    try:
        with verify_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar_one()
    finally:
        verify_engine.dispose()

    assert count == 1


def test_raw_sql_requires_explicit_allow_flag():
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        with pytest.raises(ValueError, match="Raw sql= execution is disabled by default"):
            facade.load(sql="SELECT 1 AS id", as_pandas=True)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM users",
        "SELECT 1 AS id; DROP TABLE users",
    ],
)
def test_raw_sql_blocks_mutating_or_multi_statement_even_with_allow_flag(sql):
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        with pytest.raises(ValueError, match="single-statement read-only SELECT/WITH"):
            facade.load(sql=sql, as_pandas=True, allow_raw_sql=True)


def test_raw_sql_policy_disabled_rejects_raw_sql_even_with_allow_flag():
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config, raw_sql_policy="disabled") as facade:
        with pytest.raises(ValueError, match="disabled by this DataGateway policy"):
            facade.load(sql="SELECT 1 AS id", as_pandas=True, allow_raw_sql=True)


def test_datacube_request_validator_blocks_suspicious_cube_and_filter_keys():
    def permissive_loader(_request):
        return pd.DataFrame({"ok": [True]})

    def strict_validator(request):
        cube_name = str(request.cube or "")
        if ".." in cube_name or "/" in cube_name:
            raise ValueError("cube name contains forbidden path tokens")
        if any(str(key).startswith("$") for key in request.filters):
            raise ValueError("operator-style filter keys are not allowed")

    contract = DatacubeContract(request_validator=strict_validator)

    with DataGateway(
        DatacubeConfig(loader=permissive_loader, contract=contract, default_cube="sales")
    ) as gateway:
        with pytest.raises(ValueError, match="Datacube contract request validation failed") as exc_info:
            gateway.load(cube="../admin", filters={"$where": "1=1"}, return_type="pandas")

    message = str(exc_info.value)
    assert "cube name contains forbidden path tokens" in message
    assert "filter_keys=['$where']" in message


def test_strict_filter_validation_rejects_unapproved_runtime_filter_field(tmp_path):
    db_path = tmp_path / "strict_filters.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, status TEXT)"))
            conn.execute(text("INSERT INTO users (status) VALUES ('active')"))
    finally:
        engine.dispose()

    config = SqlDatabaseConfig(
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
    )

    with DataGateway(
        config,
        table="users",
        strict_filter_validation=True,
        allowed_filter_fields={"status"},
    ) as gateway:
        with pytest.raises(ValueError, match="Filter field 'id' is not allowed"):
            gateway.load(id__exact=1, return_type="pandas")


def test_require_datacube_request_validator_enforces_contract():
    def loader(_request):
        return pd.DataFrame({"ok": [True]})

    with pytest.raises(
        ValueError,
        match="require_datacube_request_validator=True requires DatacubeContract",
    ):
        DataGateway(
            DatacubeConfig(loader=loader, default_cube="sales"),
            require_datacube_request_validator=True,
        )


# ---------------------------------------------------------------------------
# ReDoS — regex pattern validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern",
    [
        "(a+)+b",        # classic nested-quantifier ReDoS
        "(a*)*b",        # nested star
        "([a-z]+)*",     # alternation of quantified group
        "([a-z]{1,5}){", # quantified group with brace quantifier
    ],
)
def test_validate_regex_pattern_rejects_dangerous_patterns(pattern):
    """Known catastrophic-backtracking patterns must raise ValueError."""
    with pytest.raises(ValueError, match="nested quantifiers"):
        validate_regex_pattern(pattern)


def test_validate_regex_pattern_rejects_overlong_patterns():
    with pytest.raises(ValueError, match="too long"):
        validate_regex_pattern("a" * 501)


def test_validate_regex_pattern_rejects_syntactically_invalid():
    with pytest.raises(ValueError, match="Invalid regex"):
        validate_regex_pattern("[unclosed")


def test_validate_regex_pattern_accepts_safe_patterns():
    validate_regex_pattern(r"^\d{4}-\d{2}-\d{2}$")
    validate_regex_pattern(r"foo|bar")
    validate_regex_pattern(r"[a-z]+")


# ---------------------------------------------------------------------------
# FieldMap strict mode — SQL injection guard via field translation
# ---------------------------------------------------------------------------


def test_field_map_strict_mode_raises_for_unknown_key():
    fm = FieldMap({"db_col": "semantic_field"}, strict=True)
    with pytest.raises(KeyError, match="unknown_key"):
        fm.to_db("unknown_key")


def test_field_map_strict_mode_passes_known_key():
    fm = FieldMap({"db_col": "semantic_field"}, strict=True)
    assert fm.to_db("semantic_field") == "db_col"


def test_field_map_non_strict_passes_unknown_key_unchanged():
    fm = FieldMap({"db_col": "semantic_field"}, strict=False)
    assert fm.to_db("unknown_key") == "unknown_key"


# ---------------------------------------------------------------------------
# Path traversal — ParquetDataResource construction-time sandbox check
# ---------------------------------------------------------------------------


def test_parquet_resource_rejects_path_traversal_at_construction(tmp_path):
    """A traversal path must be rejected at construction, not at load time."""
    from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource

    # Use an absolute path outside the project root and temp dir to ensure the
    # sandbox boundary is actually crossed.
    traversal_path = "/etc/passwd"
    config = ParquetDataConfig(parquet_storage_path=traversal_path)
    with pytest.raises(PermissionError):
        ParquetDataResource(config)


def test_parquet_resource_rejects_null_byte_in_path(tmp_path):
    from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource

    config = ParquetDataConfig(parquet_storage_path=str(tmp_path))
    resource = ParquetDataResource(config)
    with pytest.raises(ValueError, match="null byte"):
        resource._secure_local_path("/tmp/file\x00.parquet")
