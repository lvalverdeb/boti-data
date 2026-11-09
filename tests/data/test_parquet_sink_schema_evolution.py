"""ParquetSink appends may widen a dataset's schema but never narrow it (PRD R12)."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pyarrow as pa
import pytest

from boti_data import ParquetDataConfig, ParquetDataResource
from boti_data.pipelines.sinks_parquet import ParquetSink, SchemaNarrowingError
from boti_data.pipelines.sinks_schema import unify_dataset_schemas


class _StubLogger:
    def debug(self, *_a, **_k) -> None:
        pass

    def info(self, *_a, **_k) -> None:
        pass

    def warning(self, *_a, **_k) -> None:
        pass

    def error(self, *_a, **_k) -> None:
        pass


def _config(root, **extra) -> ParquetDataConfig:
    return ParquetDataConfig(
        logger=_StubLogger(),
        parquet_storage_path=str(root),
        partition_on=["partition_date"],
        **extra,
    )


def _write(root, frame: pd.DataFrame, *, overwrite: bool) -> None:
    with ParquetSink(_config(root)) as sink:
        sink.write(frame, date_field="event_date", overwrite=overwrite)


def _first_batch() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": pd.Series([1, 2], dtype="int32"),
            "amount": pd.Series([1.5, 2.5], dtype="float32"),
            "event_date": [dt.date(2026, 1, 1), dt.date(2026, 1, 1)],
        }
    )


def test_append_with_new_column_and_promoted_types_reads_back_end_to_end(temp_project_root) -> None:
    root = temp_project_root / "evolving"
    _write(root, _first_batch(), overwrite=True)
    second = pd.DataFrame(
        {
            "id": pd.Series([3], dtype="int64"),
            "amount": pd.Series([3.5], dtype="float64"),
            "note": ["late"],
            "event_date": [dt.date(2026, 1, 2)],
        }
    )
    _write(root, second, overwrite=False)

    config = _config(
        root, parquet_start_date=dt.date(2026, 1, 1), parquet_end_date=dt.date(2026, 1, 2)
    )
    with ParquetDataResource(config) as resource:
        frame = resource.load_files().compute().sort_values("id").reset_index(drop=True)
        table = resource.load_arrow().sort_by("id")

    assert frame["id"].tolist() == [1, 2, 3]
    assert frame["amount"].tolist() == [1.5, 2.5, 3.5]
    assert frame["note"].tolist()[2] == "late" and frame["note"].isna().tolist()[:2] == [True, True]
    assert frame["partition_date"].astype(str).tolist() == [
        "2026-01-01",
        "2026-01-01",
        "2026-01-02",
    ]
    assert str(frame["id"].dtype) == "int64"
    assert table.schema.field("id").type == pa.int64()
    assert table.schema.field("amount").type == pa.float64()
    assert table.column("note").to_pylist() == [None, None, "late"]


def test_append_with_filters_and_columns_on_evolved_dataset(temp_project_root) -> None:
    root = temp_project_root / "evolving_filtered"
    _write(root, _first_batch(), overwrite=True)
    second = pd.DataFrame(
        {"id": [3, 4], "note": ["a", "b"], "event_date": [dt.date(2026, 1, 2)] * 2}
    )
    _write(root, second, overwrite=False)

    config = _config(
        root, parquet_start_date=dt.date(2026, 1, 1), parquet_end_date=dt.date(2026, 1, 2)
    )
    with ParquetDataResource(config) as resource:
        frame = resource.load_files(filters=[("id", ">", 2)], columns=["id", "note"]).compute()
    assert sorted(frame.columns) == ["id", "note"]
    assert sorted(frame["id"].tolist()) == [3, 4]


def test_narrowing_append_is_rejected_before_writing(temp_project_root) -> None:
    root = temp_project_root / "narrowing"
    _write(root, _first_batch(), overwrite=True)
    files_before = sorted(p.name for p in root.rglob("*.parquet"))
    bad = pd.DataFrame({"id": ["not-a-number"], "event_date": [dt.date(2026, 1, 3)]})

    with pytest.raises(
        SchemaNarrowingError, match="column 'id' from int32 to (large_)?string"
    ) as info:
        _write(root, bad, overwrite=False)

    assert info.value.column == "id"
    assert sorted(p.name for p in root.rglob("*.parquet")) == files_before


def test_overwrite_ignores_the_existing_schema(temp_project_root) -> None:
    root = temp_project_root / "overwritten"
    _write(root, _first_batch(), overwrite=True)
    replacement = pd.DataFrame({"id": ["text-now"], "event_date": [dt.date(2026, 1, 3)]})
    _write(root, replacement, overwrite=True)
    config = _config(root)
    with ParquetDataResource(config) as resource:
        assert resource.load_files().compute()["id"].tolist() == ["text-now"]


def test_unify_dataset_schemas_names_the_conflicting_column() -> None:
    existing = pa.schema([("a", pa.int64()), ("b", pa.string())])
    incoming = pa.schema([("a", pa.int32()), ("b", pa.int64())])
    with pytest.raises(SchemaNarrowingError) as info:
        unify_dataset_schemas([existing, incoming])
    assert (info.value.column, info.value.existing, info.value.incoming) == (
        "b",
        pa.string(),
        pa.int64(),
    )
    widened = unify_dataset_schemas(
        [pa.schema([("a", pa.int32())]), pa.schema([("a", pa.float64())])]
    )
    assert widened.field("a").type == pa.float64()


def test_files_with_incompatible_types_still_read_like_before(tmp_path) -> None:
    """string in one file, int64 in another is not a widening: the plain read applies."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from boti_data.parquet import ParquetDataConfig, ParquetDataResource

    root = tmp_path / "ds"
    root.mkdir()
    pq.write_table(
        pa.table({"id": [1, 2], "card": pa.array([11, 12], pa.int64())}), root / "a.parquet"
    )
    pq.write_table(pa.table({"id": [3, 4], "card": pa.array(["13", "14"])}), root / "b.parquet")
    with ParquetDataResource(ParquetDataConfig(parquet_storage_path=str(root))) as resource:
        frame = resource.load_arrow().to_pandas()
    assert sorted(frame["id"].tolist()) == [1, 2, 3, 4]
