"""Regressions from the 2026-11 security audit, medium findings."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from sqlalchemy.pool import NullPool

from boti_data.db import engine_registry, sql_engine
from boti_data.db.engine_registry import EngineRegistry
from boti_data.db.partitioned_types import SqlPartitionSpec
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.filters.backend_ops import operation_map_dask, operation_map_sqlalchemy
from boti_data.pipelines.sinks import CsvSink, ParquetSink
from boti_data.pipelines.sinks_schema import SchemaNarrowingError

pytestmark = pytest.mark.security_regression


@pytest.fixture
def project_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    return root


# --- text sinks: remote targets go through FilesystemConfig's endpoint validation --------


def test_csv_sink_refuses_a_loopback_http_target_before_any_request(project_root) -> None:
    sink = CsvSink(
        {"storage_path": "http://127.0.0.1:9/latest/meta-data", "project_root": str(project_root)}
    )
    with sink, pytest.raises(ValueError, match="(?i)endpoint|private|reserved|loopback"):
        sink.write(pd.DataFrame({"x": [1]}))


# --- Parquet sink writes where the resource reads: the project root ---------------------


def test_parquet_sink_relative_path_writes_under_the_project_root(
    project_root, tmp_path, monkeypatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    sink = ParquetSink(
        {"parquet_storage_path": "rel_out", "project_root": str(project_root)}, partition_on=None
    )
    with sink:
        sink.write(pd.DataFrame({"k": [1]}))
        assert sink.reader.resource.load_files().compute()["k"].tolist() == [1]
    assert (project_root / "rel_out").exists()
    assert not (elsewhere / "rel_out").exists()


# --- engine identity includes connect_args and execution_options -------------------------


def test_engine_keys_differ_on_connect_args() -> None:
    plain = SqlDatabaseConfig(
        connection_url="postgresql+psycopg://u@db/app", connect_args={"sslmode": "disable"}
    )
    verified = SqlDatabaseConfig(
        connection_url="postgresql+psycopg://u@db/app",
        connect_args={"sslmode": "verify-full", "sslrootcert": "/etc/ca.pem"},
    )
    assert sql_engine._build_sync_engine_key(plain) != sql_engine._build_sync_engine_key(verified)
    assert sql_engine._build_async_engine_key(plain) != sql_engine._build_async_engine_key(verified)
    assert sql_engine._get_worker_engine_identity(plain) != sql_engine._get_worker_engine_identity(
        verified
    )


# --- the read-only hook runs before an engine is visible to other holders ----------------


class _FakeEngine:
    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


def test_registry_configures_an_engine_before_registering_it(monkeypatch) -> None:
    key = ("test", "configure-before-register", id(object()))
    seen: list[object] = []
    monkeypatch.setattr(engine_registry, "create_engine", lambda url, **kw: _FakeEngine())
    monkeypatch.setattr(engine_registry, "instrument_engine_pool", lambda engine, url: None)

    def configure(engine: object) -> None:
        seen.append(EngineRegistry._check_registered(key))

    engine, reused = EngineRegistry.get_or_create(key, "sqlite://", configure=configure)
    try:
        assert reused is False and seen == [None]
        assert EngineRegistry._check_registered(key) is engine
    finally:
        EngineRegistry._decrement_and_maybe_pop(key)


# --- filters: contains is literal on both backends; regex patterns are validated --------


def test_dask_contains_is_literal_like_the_sql_twin() -> None:
    series = pd.Series(["abc", "a.c", None])
    mask = operation_map_dask()["contains"](series, "a.c")
    assert mask.tolist() == [False, True, False]


@pytest.mark.parametrize("operation", ["regex", "iregex"])
def test_regex_filters_reject_nested_quantifiers_on_both_backends(operation) -> None:
    from sqlalchemy import Column, String

    evil = "(a+)+$"
    with pytest.raises(ValueError, match="(?i)nested quantifiers"):
        operation_map_dask()[operation](pd.Series(["aaaa"]), evil)
    with pytest.raises(ValueError, match="(?i)nested quantifiers"):
        operation_map_sqlalchemy()[operation](Column("name", String), evil)


# --- append guard sees every file, not the first one ------------------------------------


def test_append_cannot_narrow_a_column_added_by_a_later_partition(project_root) -> None:
    sink = ParquetSink(
        {"parquet_storage_path": str(project_root / "evo"), "project_root": str(project_root)},
        partition_on=["partition_date"],
    )
    with sink:
        sink.write(pd.DataFrame({"x": [1], "partition_date": ["2024-01-01"]}))
        sink.write(
            pd.DataFrame({"x": [2], "y": ["a"], "partition_date": ["2024-01-02"]}),
            overwrite=False,
        )
        with pytest.raises(SchemaNarrowingError):
            sink.write(
                pd.DataFrame({"x": [3], "y": [7], "partition_date": ["2024-01-03"]}),
                overwrite=False,
            )


# --- checkpoint keys survive the JSON round trip -----------------------------------------


def test_partition_key_is_stable_across_checkpoint_round_trip() -> None:
    spec = SqlPartitionSpec(sql="select 1 where d >= ?", params=(dt.date(2024, 1, 1),))
    assert SqlPartitionSpec.from_json(spec.to_json()).key() == spec.key()
    named = SqlPartitionSpec(sql="select 1", params={"b": 2, "a": dt.date(2024, 1, 1)})
    assert SqlPartitionSpec.from_json(named.to_json()).key() == named.key()


def test_null_pool_engines_still_share_a_key_for_equal_configs() -> None:
    config = SqlDatabaseConfig(connection_url="sqlite://", poolclass="sqlalchemy.pool.NullPool")
    same = SqlDatabaseConfig(connection_url="sqlite://", poolclass="sqlalchemy.pool.NullPool")
    assert config.poolclass is NullPool
    assert sql_engine._build_sync_engine_key(config) == sql_engine._build_sync_engine_key(same)
