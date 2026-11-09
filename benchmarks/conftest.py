"""Session-scoped inputs shared by every benchmark case."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmarks._seed import ROWS_1M, seed_parquet, seed_sqlite

DSN_ENV_VAR = "BOTI_BENCH_DB_DSN"


@pytest.fixture(scope="session")
def bench_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("boti-bench")
    (root / "pyproject.toml").touch()  # a project root for sandboxed Parquet paths
    return root


@pytest.fixture(scope="session")
def sqlite_1m_dsn(bench_root: Path) -> str:
    dsn = seed_sqlite(bench_root / "events.db", ROWS_1M)
    os.environ[DSN_ENV_VAR] = dsn  # worker processes read the DSN from here
    return dsn


@pytest.fixture(scope="session")
def parquet_1m_root(bench_root: Path) -> Path:
    return seed_parquet(bench_root / "events_parquet", ROWS_1M)


@pytest.fixture
def sql_helper(sqlite_1m_dsn: str):
    from boti_data import DataHelper

    helper = DataHelper(
        backend="sqlalchemy",
        connection_url=sqlite_1m_dsn,
        worker_connection_env_var=DSN_ENV_VAR,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="events",
    )
    try:
        yield helper
    finally:
        helper.close()
