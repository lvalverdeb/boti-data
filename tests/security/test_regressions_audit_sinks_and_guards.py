"""Regressions from the 2026-11 security audit: sink paths, sandbox schemes, read-only guards."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import create_engine, text

from boti_data.db import sql_engine
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.gateway.sql_guard import is_read_only_sql
from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource
from boti_data.pipelines.sinks import CsvSink, JsonlSink, ParquetSink

pytestmark = pytest.mark.security_regression


@pytest.fixture
def project_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    return root


# --- partition values from data cannot leave the sink directory -------------------------

_ESCAPE = "../../../../escaped"


@pytest.mark.parametrize("unsafe", [_ESCAPE, "a/b", "..", ".", "", "nul\x00byte", "back\\slash"])
def test_parquet_sink_rejects_partition_values_that_cannot_name_a_directory(
    project_root, tmp_path, unsafe
) -> None:
    sink = ParquetSink(
        {"parquet_storage_path": str(project_root / "out"), "project_root": str(project_root)},
        partition_on=["region"],
    )
    with sink, pytest.raises(ValueError, match="cannot name a directory"):
        sink.write(pd.DataFrame({"k": [1, 2], "region": ["eu", unsafe]}))
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("sink_type", [CsvSink, JsonlSink])
def test_text_sinks_reject_partition_values_that_cannot_name_a_directory(
    project_root, tmp_path, sink_type
) -> None:
    config = {
        "storage_path": str(project_root / "out"),
        "partition_on": ["partition_date"],
        "project_root": str(project_root),
    }
    with sink_type(config) as sink, pytest.raises(ValueError, match="cannot name a directory"):
        sink.write(pd.DataFrame({"k": [1], "partition_date": [_ESCAPE]}))
    assert not (tmp_path / "escaped").exists()


def test_ordinary_partition_values_still_write(project_root) -> None:
    sink = ParquetSink(
        {"parquet_storage_path": str(project_root / "out"), "project_root": str(project_root)},
        partition_on=["region"],
    )
    with sink:
        sink.write(pd.DataFrame({"k": [1, 2], "region": ["eu-west 1", "us=east"]}))
        frame = sink.reader.resource.load_files().compute()
    assert sorted(frame["k"].tolist()) == [1, 2]


# --- a scheme prefix cannot bypass the sandbox on a local filesystem --------------------


@pytest.mark.parametrize("scheme", ["memory://", "s3://", "gs://"])
def test_scheme_prefixed_path_on_local_filesystem_is_denied(project_root, tmp_path, scheme) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    pq.write_table(pa.table({"secret": ["TOPSECRET"]}), outside / "data.parquet")
    config = ParquetDataConfig(
        parquet_storage_path=f"{scheme}{outside}", project_root=str(project_root), partition_on=None
    )
    with ParquetDataResource(config) as resource:
        with pytest.raises(PermissionError, match="filesystem is local"):
            resource.load_files().compute()
        with pytest.raises(PermissionError, match="filesystem is local"):
            resource.load_arrow()


# --- overwrite=False appends instead of replacing the previous part files ---------------


def test_parquet_append_keeps_earlier_files(project_root) -> None:
    sink = ParquetSink(
        {"parquet_storage_path": str(project_root / "flat"), "project_root": str(project_root)},
        partition_on=None,
    )
    with sink:
        sink.write(pd.DataFrame({"x": [1, 2]}))
        sink.write(pd.DataFrame({"x": [3]}), overwrite=False)
        sink.write(pd.DataFrame({"x": [4]}), overwrite=False)
        rows = sink.reader.resource.load_files().compute()["x"].tolist()
    assert sorted(rows) == [1, 2, 3, 4]
    assert len(list((project_root / "flat").glob("*.parquet"))) == 3


def test_parquet_append_into_an_existing_partition_keeps_earlier_files(project_root) -> None:
    sink = ParquetSink(
        {"parquet_storage_path": str(project_root / "part"), "project_root": str(project_root)},
        partition_on=["partition_date"],
    )
    with sink:
        sink.write(pd.DataFrame({"x": [1, 2], "partition_date": ["2024-01-01"] * 2}))
        sink.write(pd.DataFrame({"x": [3], "partition_date": ["2024-01-01"]}), overwrite=False)
        rows = sink.reader.resource.load_files().compute()["x"].tolist()
    assert sorted(rows) == [1, 2, 3]


@pytest.mark.parametrize(("sink_type", "suffix"), [(CsvSink, "csv"), (JsonlSink, "jsonl")])
def test_text_sink_append_keeps_earlier_files(project_root, sink_type, suffix) -> None:
    config = {"storage_path": str(project_root / "out"), "project_root": str(project_root)}
    with sink_type(config) as sink:
        sink.write(pd.DataFrame({"x": [1, 2]}))
        sink.write(pd.DataFrame({"x": [3]}), overwrite=False)
    assert len(list((project_root / "out").glob(f"*.{suffix}"))) == 2


# --- raw SQL guard and DuckDB external access -------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * INTO copied FROM t",
        "SELECT set_config('x', 'y', false)",
        "SET search_path = evil",
        "SELECT * FROM t; LOAD httpfs",
        "INSTALL httpfs",
        "SELECT lo_import('/etc/passwd')",
        "SELECT setval('seq', 1)",
        "SELECT pg_terminate_backend(1)",
        "WITH x AS (SELECT 1) SELECT * FROM x INTO y",
    ],
)
def test_raw_sql_guard_rejects_statements_with_side_effects(sql) -> None:
    assert not is_read_only_sql(sql)


def test_raw_sql_guard_keeps_plain_selects() -> None:
    assert is_read_only_sql("SELECT id, name FROM users WHERE offset_col > 1 ORDER BY id OFFSET 5")


def test_duckdb_query_only_blocks_file_and_network_reads(tmp_path) -> None:
    duckdb = pytest.importorskip("duckdb")
    db_path = tmp_path / "guard.duckdb"
    duckdb.connect(str(db_path)).execute("create table t as select 1 as x").close()
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET\n")
    config = SqlDatabaseConfig(connection_url=f"duckdb:///{db_path}", query_only=True)
    engine = create_engine(
        config.connection_url.get_secret_value(),
        **sql_engine._build_sync_engine_kwargs(config),
    )
    try:
        with engine.connect() as conn:
            assert conn.execute(text("select x from t")).scalar() == 1
            with pytest.raises(Exception, match="(?i)permission|external access"):
                conn.execute(text(f"select content from read_text('{secret}')")).scalar()
            with pytest.raises(Exception):
                conn.execute(text("SET enable_external_access = true"))
    finally:
        engine.dispose()


# --- async worker engines get the same read-only session statement ----------------------


def test_worker_async_engine_configures_query_only(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "worker.db"
    create_engine(f"sqlite:///{db_path}").dispose()
    configured: list[object] = []
    monkeypatch.setattr(
        sql_engine,
        "_configure_query_only_engine",
        lambda engine, parsed: configured.append((engine, parsed.get_backend_name())),
    )
    config = SqlDatabaseConfig(
        connection_url=f"sqlite+aiosqlite:///file:{db_path}?mode=ro&uri=true", query_only=True
    )
    engine = sql_engine._create_worker_async_engine(config)
    try:
        assert configured == [(engine.sync_engine, "sqlite")]
    finally:
        import asyncio

        asyncio.run(engine.dispose())
