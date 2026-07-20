"""
Security regression tests: raw-SQL policy gating and datacube/filter
request validation.

Split out of test_regressions.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from boti_data import DatacubeConfig, DatacubeContract
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.gateway import DataGateway

pytestmark = pytest.mark.security_regression


@pytest.mark.parametrize(
    "payload",
    [
        "active'; DROP TABLE users; --",
        "active' OR 1=1 --",
        "active' UNION SELECT 1, 'x' --",
        "active' /* inline */ OR '1'='1",
    ],
)
def test_raw_sql_params_do_not_allow_statement_injection(tmp_path, payload) -> None:
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


def test_raw_sql_requires_explicit_allow_flag() -> None:
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
def test_raw_sql_blocks_mutating_or_multi_statement_even_with_allow_flag(sql) -> None:
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config) as facade:
        with pytest.raises(ValueError, match="single-statement read-only SELECT/WITH"):
            facade.load(sql=sql, as_pandas=True, allow_raw_sql=True)


def test_raw_sql_policy_disabled_rejects_raw_sql_even_with_allow_flag() -> None:
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with DataGateway(config, raw_sql_policy="disabled") as facade:
        with pytest.raises(ValueError, match="disabled by this DataGateway policy"):
            facade.load(sql="SELECT 1 AS id", as_pandas=True, allow_raw_sql=True)


def test_datacube_request_validator_blocks_suspicious_cube_and_filter_keys() -> None:
    def permissive_loader(_request) -> pd.DataFrame:
        return pd.DataFrame({"ok": [True]})

    def strict_validator(request) -> None:
        cube_name = str(request.cube or "")
        if ".." in cube_name or "/" in cube_name:
            raise ValueError("cube name contains forbidden path tokens")
        if any(str(key).startswith("$") for key in request.filters):
            raise ValueError("operator-style filter keys are not allowed")

    contract = DatacubeContract(request_validator=strict_validator)

    with DataGateway(
        DatacubeConfig(loader=permissive_loader, contract=contract, default_cube="sales")
    ) as gateway:
        with pytest.raises(
            ValueError, match="Datacube contract request validation failed"
        ) as exc_info:
            gateway.load(cube="../admin", filters={"$where": "1=1"}, return_type="pandas")

    message = str(exc_info.value)
    assert "cube name contains forbidden path tokens" in message
    assert "filter_keys=['$where']" in message


def test_strict_filter_validation_rejects_unapproved_runtime_filter_field(tmp_path) -> None:
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


def test_require_datacube_request_validator_enforces_contract() -> None:
    def loader(_request) -> pd.DataFrame:
        return pd.DataFrame({"ok": [True]})

    with pytest.raises(
        ValueError,
        match="require_datacube_request_validator=True requires DatacubeContract",
    ):
        DataGateway(
            DatacubeConfig(loader=loader, default_cube="sales"),
            require_datacube_request_validator=True,
        )
