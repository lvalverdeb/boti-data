"""
Parquet-backed DataHelper and ParquetReader tests: engine-view defaults over
parquet, partition-date hive pruning, and ParquetReader's sync/async load
paths (filters, in-filters, legacy bare-filter kwargs, and return types).

Split out of test_helper.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.fs as pafs
import pytest

from boti_data import DataHelper, ParquetReader


def test_helper_parquet_polars_view_defaults_to_eager_polars(temp_project_root) -> None:
    file_path = temp_project_root / "helper_engine_views" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(
        file_path, index=False
    )

    helper = DataHelper(
        backend="parquet",
        storage_path=str(file_path.parent),
        parquet_filename="users",
    )
    try:
        frame = helper.polars.load(filters={"status__exact": "active"})
    finally:
        helper.close()

    assert isinstance(frame, pl.DataFrame)
    assert frame["id"].to_list() == [1]


def test_helper_from_legacy_config_defaults_to_partition_date_hive_pruning(
    temp_project_root,
) -> None:
    root = temp_project_root / "legacy_hive_partitioned"
    (root / "partition_date=2026-01-01").mkdir(parents=True)
    (root / "partition_date=2026-04-01").mkdir(parents=True)
    pd.DataFrame({"id": [1], "status": ["in_range"]}).to_parquet(
        root / "partition_date=2026-01-01" / "part1.parquet",
        index=False,
    )
    pd.DataFrame({"id": [2], "status": ["out_of_range"]}).to_parquet(
        root / "partition_date=2026-04-01" / "part2.parquet",
        index=False,
    )

    helper = DataHelper.from_legacy_config(
        {
            "backend": "parquet",
            "storage_path": str(root),
            "parquet_start_date": "2026-01-01",
            "parquet_end_date": "2026-03-31",
        }
    )
    try:
        frame = helper.load(as_pandas=True)
    finally:
        helper.close()

    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [1]
    assert frame["status"].tolist() == ["in_range"]


def test_parquet_reader_defaults_to_partition_date_and_configured_eager_pandas(
    temp_project_root,
) -> None:
    file_path = temp_project_root / "reader_data" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(
        file_path, index=False
    )

    reader = ParquetReader(
        storage_path=str(file_path.parent),
        parquet_filename="users",
        return_type="pandas",
        execution_mode="eager",
    )
    try:
        frame = reader.load()
    finally:
        reader.close()

    assert reader.partition_on == ["partition_date"]
    assert reader.parquet_storage_path == str(file_path.parent)
    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == [1, 2]


def test_parquet_reader_runtime_return_type_override_supports_lazy_dask(temp_project_root) -> None:
    file_path = temp_project_root / "reader_lazy" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(
        file_path, index=False
    )

    reader = ParquetReader(storage_path=str(file_path.parent), parquet_filename="users")
    try:
        frame = reader.load(return_type="dask", execution_mode="lazy")
    finally:
        reader.close()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["status"].tolist() == ["active", "inactive"]


def test_parquet_reader_load_applies_filters_in_lazy_mode(temp_project_root) -> None:
    file_path = temp_project_root / "reader_filters" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "status": ["active", "inactive", "active"]}).to_parquet(
        file_path,
        index=False,
    )

    reader = ParquetReader(storage_path=str(file_path.parent), parquet_filename="users")
    try:
        frame = reader.load(filters={"status__exact": "active"})
    finally:
        reader.close()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1, 3]


def test_parquet_reader_load_applies_in_filters_in_lazy_mode(temp_project_root) -> None:
    file_path = temp_project_root / "reader_in_filters" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"associate_id": [2819, 4499, 7777], "status": ["a", "b", "c"]}).to_parquet(
        file_path,
        index=False,
    )

    reader = ParquetReader(storage_path=str(file_path.parent), parquet_filename="users")
    try:
        frame = reader.load(filters={"associate_id__in": [2819, 4499]})
    finally:
        reader.close()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["associate_id"].tolist() == [2819, 4499]


def test_parquet_reader_load_supports_legacy_bare_filter_kwargs(temp_project_root) -> None:
    file_path = temp_project_root / "reader_bare_filters" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"associate_id": [2819, 4499, 7777], "status": ["a", "b", "c"]}).to_parquet(
        file_path,
        index=False,
    )

    reader = ParquetReader(storage_path=str(file_path.parent), parquet_filename="users")
    try:
        frame = reader.load(associate_id__in=[2819, 4499])
    finally:
        reader.close()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["associate_id"].tolist() == [2819, 4499]


@pytest.mark.asyncio
async def test_parquet_reader_aload_supports_eager_arrow_return_type(temp_project_root) -> None:
    file_path = temp_project_root / "reader_arrow" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1], "status": ["active"]}).to_parquet(file_path, index=False)

    reader = ParquetReader(storage_path=str(file_path.parent), parquet_filename="users")
    try:
        frame = await reader.aload(return_type="arrow", execution_mode="eager")
    finally:
        await reader.aclose()

    assert isinstance(frame, pa.Table)
    assert frame.to_pydict()["status"] == ["active"]


@pytest.mark.asyncio
async def test_parquet_reader_aload_applies_filters_with_explicit_pyarrow_fs(
    temp_project_root,
) -> None:
    file_path = temp_project_root / "reader_filter_pyarrow_fs" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(
        file_path, index=False
    )

    reader = ParquetReader(
        storage_path=str(file_path.parent),
        parquet_filename="users",
        fs=pafs.LocalFileSystem(),
    )
    try:
        frame = await reader.aload(filters={"status__exact": "active"})
    finally:
        await reader.aclose()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1]


@pytest.mark.asyncio
async def test_parquet_reader_aload_applies_in_filters_with_explicit_pyarrow_fs(
    temp_project_root,
) -> None:
    file_path = temp_project_root / "reader_in_filter_pyarrow_fs" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"associate_id": [2819, 4499, 7777], "status": ["a", "b", "c"]}).to_parquet(
        file_path,
        index=False,
    )

    reader = ParquetReader(
        storage_path=str(file_path.parent),
        parquet_filename="users",
        fs=pafs.LocalFileSystem(),
    )
    try:
        frame = await reader.aload(filters={"associate_id__in": [2819, 4499]})
    finally:
        await reader.aclose()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["associate_id"].tolist() == [2819, 4499]


@pytest.mark.asyncio
async def test_parquet_reader_aload_supports_legacy_bare_filter_kwargs(temp_project_root) -> None:
    file_path = temp_project_root / "reader_bare_async_filters" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"associate_id": [2819, 4499, 7777], "status": ["a", "b", "c"]}).to_parquet(
        file_path,
        index=False,
    )

    reader = ParquetReader(storage_path=str(file_path.parent), parquet_filename="users")
    try:
        frame = await reader.aload(associate_id__in=[2819, 4499])
    finally:
        await reader.aclose()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["associate_id"].tolist() == [2819, 4499]


@pytest.mark.asyncio
async def test_parquet_reader_aload_supports_lazy_dask_with_explicit_pyarrow_fs(
    temp_project_root,
) -> None:
    file_path = temp_project_root / "reader_pyarrow_fs" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(
        file_path, index=False
    )

    reader = ParquetReader(
        storage_path=str(file_path.parent),
        parquet_filename="users",
        fs=pafs.LocalFileSystem(),
    )
    try:
        frame = await reader.aload()
    finally:
        await reader.aclose()

    assert isinstance(frame, dd.DataFrame)
    assert frame.compute()["id"].tolist() == [1, 2]


@pytest.mark.asyncio
async def test_parquet_reader_aload_applies_partition_filters_for_hive_layout(
    temp_project_root,
) -> None:
    root = temp_project_root / "reader_partition_filters"
    (root / "partition_date=2026-01-01").mkdir(parents=True)
    (root / "partition_date=2026-01-02").mkdir(parents=True)
    pd.DataFrame({"id": [1], "status": ["active"]}).to_parquet(
        root / "partition_date=2026-01-01" / "part1.parquet",
        index=False,
    )
    pd.DataFrame({"id": [2], "status": ["inactive"]}).to_parquet(
        root / "partition_date=2026-01-02" / "part2.parquet",
        index=False,
    )

    reader = ParquetReader(storage_path=str(root))
    try:
        frame = await reader.aload(filters={"partition_date__exact": "2026-01-01"})
    finally:
        await reader.aclose()

    assert isinstance(frame, dd.DataFrame)
    computed = frame.compute().sort_values("id").reset_index(drop=True)
    assert computed["id"].tolist() == [1]
    assert computed["partition_date"].tolist() == ["2026-01-01"]
