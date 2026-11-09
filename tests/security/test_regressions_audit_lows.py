"""Regressions from the 2026-11 security audit, low findings."""

from __future__ import annotations

import datetime as dt
import decimal

import pandas as pd
import pytest
from sqlalchemy import Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data.db.partitioned_types import SqlPartitionSpec
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.db.sql_repository import SqlRepository
from boti_data.gateway.chunking import MAX_IN_CHUNK_CONCURRENCY, InChunkPlanner
from boti_data.gateway.sql_strategy import SqlAlchemyStrategy
from boti_data.parquet import schema_filters
from boti_data.pipelines.sinks import ParquetSink
from boti_data.pipelines.sinks_common import _write_with_staging

pytestmark = pytest.mark.security_regression


class Base(DeclarativeBase):
    pass


class Person(Base):
    __tablename__ = "people"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    ssn: Mapped[str] = mapped_column(String(20))


@pytest.fixture
def project_root(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    return root


# --- staging: per-run directory, bucket roots refused ------------------------------------


def test_two_overwriters_do_not_share_a_staging_directory(project_root) -> None:
    import fsspec

    fs = fsspec.filesystem("file")
    target = str(project_root / "target")
    fs.makedirs(target, exist_ok=True)
    (project_root / "target" / "old.txt").write_text("old")
    staged: list[str] = []

    def write_fn(directory: str) -> list[str]:
        staged.append(directory)
        fs.makedirs(directory, exist_ok=True)
        (project_root / f"{directory.rsplit('/', 1)[1]}" / "new.txt").write_text("new")
        return [f"{directory}/new.txt"]

    _write_with_staging(fs=fs, target_path=target, overwrite=True, write_fn=write_fn)
    _write_with_staging(fs=fs, target_path=target, overwrite=True, write_fn=write_fn)
    assert len(set(staged)) == 2 and all(".staging-" in path for path in staged)
    assert not any(path.endswith(".staging") for path in staged)


def test_overwriting_a_bucket_root_is_refused() -> None:
    import fsspec

    fs = fsspec.filesystem("memory")
    fs.makedirs("memory://bucket", exist_ok=True)
    fs.pipe("memory://bucket/x.txt", b"x")
    with pytest.raises(ValueError, match="bucket root"):
        _write_with_staging(
            fs=fs, target_path="memory://bucket", overwrite=True, write_fn=lambda d: []
        )


# --- evolved schema read once per file list ----------------------------------------------


def test_evolved_schema_footers_are_read_once_per_file_list(project_root, monkeypatch) -> None:
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
        resource = sink.reader.resource
        calls: list[str] = []
        real = schema_filters.pq.read_schema
        monkeypatch.setattr(
            schema_filters.pq, "read_schema", lambda *a, **k: (calls.append("r"), real(*a, **k))[1]
        )
        first = resource.load_files().compute()
        second = resource.load_files().compute()
    assert len(first) == len(second) == 2
    assert len(calls) == 2  # two files, read once each, not again on the second load


# --- checkpoints restore typed parameters --------------------------------------------------


def test_checkpoint_round_trip_restores_parameter_types() -> None:
    params = (dt.date(2024, 1, 1), dt.datetime(2024, 1, 1, 12, 30), decimal.Decimal("1.50"))
    spec = SqlPartitionSpec(sql="select 1", params=params)
    restored = SqlPartitionSpec.from_json(spec.to_json())
    assert restored.params == params
    assert [type(value) for value in restored.params] == [type(value) for value in params]
    assert restored.key() == spec.key()
    named = SqlPartitionSpec(sql="select 1", params={"d": dt.date(2024, 1, 1), "n": 2})
    assert SqlPartitionSpec.from_json(named.to_json()).params == named.params


# --- columns= cannot widen the projection ---------------------------------------------------


def test_columns_are_resolved_through_the_mapper() -> None:
    options = {"columns": ["__table__"], "statement": select(Person), "model": Person}
    with pytest.raises(AttributeError, match="not found in model"):
        SqlAlchemyStrategy._project_through_field_map(options, None)


# --- fan-out knobs are bounded ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "concurrency"),
    [(0, None), (-5, None), (10, 0), (10, MAX_IN_CHUNK_CONCURRENCY + 1), (10, 100_000)],
)
def test_in_chunk_controls_are_bounded(size, concurrency) -> None:
    with pytest.raises(ValueError, match="in_chunk_"):
        InChunkPlanner._explicit_controls(size, concurrency)


# --- repository update assigns mapped, non-key columns only ---------------------------------


@pytest.fixture
def people_repository(tmp_path):
    db = tmp_path / "people.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Person(id=1, name="Ann", ssn="000"))
        session.commit()
    engine.dispose()
    config = SqlDatabaseConfig(connection_url=f"sqlite:///{db}", query_only=False)
    with SqlRepository(config, "people", query_only=False) as repository:
        yield repository


def test_repository_update_rejects_unmapped_keys_and_primary_keys(people_repository) -> None:
    with pytest.raises(ValueError, match="not mapped columns"):
        people_repository.update(1, {"name": "Bo", "__class__": object})
    with pytest.raises(ValueError, match="primary key"):
        people_repository.update(1, {"id": 2})
    assert people_repository.update(1, {"name": "Bo"})["name"] == "Bo"
