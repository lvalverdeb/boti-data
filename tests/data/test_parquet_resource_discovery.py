"""
Parquet resource tests: dataset discovery, hive-partition pruning, recency
checks, and metadata/security guards.

Split out of test_parquet_resource.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pytest

from boti_data import ParquetDataConfig, ParquetDataResource


class StubLogger:
    def debug(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def error(self, *_args, **_kwargs) -> None:
        pass


def test_parquet_discover_all_files(temp_project_root) -> None:
    root = temp_project_root / "scan_data"
    (root / "nested").mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(root / "part1.parquet", index=False)
    pd.DataFrame({"id": [2]}).to_parquet(root / "nested" / "part2.parquet", index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(root),
    )

    with ParquetDataResource(config) as resource:
        discovered = sorted(Path(path).name for path in resource._discover_all_files())

    assert discovered == ["part1.parquet", "part2.parquet"]


def test_parquet_discover_all_files_raises_on_non_missing_scan_error(
    temp_project_root, monkeypatch
) -> None:
    root = temp_project_root / "scan_failure"
    root.mkdir(parents=True)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(root),
    )

    def dataset_with_permission_error(*_args, **_kwargs) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(ds, "dataset", dataset_with_permission_error)

    with ParquetDataResource(config) as resource:
        with pytest.raises(RuntimeError, match="Failed to scan parquet dataset"):
            resource._discover_all_files()


def test_parquet_partitioned_directory_discovery(temp_project_root) -> None:
    root = temp_project_root / "partitioned"
    (root / "2024" / "1" / "1").mkdir(parents=True)
    (root / "2025" / "1" / "1").mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(
        root / "2024" / "1" / "1" / "part_2024.parquet", index=False
    )
    pd.DataFrame({"id": [2]}).to_parquet(
        root / "2025" / "1" / "1" / "part_2025.parquet", index=False
    )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(root),
        parquet_start_date=dt.date(2024, 1, 1),
        parquet_end_date=dt.date(2024, 12, 31),
    )

    with ParquetDataResource(config) as resource:
        discovered = resource._discover_partitioned_files()

    assert len(discovered) == 1
    assert discovered[0].endswith("part_2024.parquet")


def test_parquet_partitioned_hive_discovery(temp_project_root) -> None:
    root = temp_project_root / "hive_partitioned"
    (root / "event_date=2024-01-01").mkdir(parents=True)
    (root / "event_date=2024-01-02").mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(
        root / "event_date=2024-01-01" / "part1.parquet", index=False
    )
    pd.DataFrame({"id": [2]}).to_parquet(
        root / "event_date=2024-01-02" / "part2.parquet", index=False
    )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(root),
        parquet_start_date=dt.date(2024, 1, 1),
        parquet_end_date=dt.date(2024, 1, 1),
        partition_on=["event_date"],
    )

    with ParquetDataResource(config) as resource:
        discovered = resource._discover_partitioned_files()

    assert len(discovered) == 1
    assert discovered[0].endswith("part1.parquet")


def test_parquet_partitioned_hive_negative_pruning(temp_project_root) -> None:
    root = temp_project_root / "hive_negative"
    for day in (1, 2, 3):
        (root / f"event_date=2024-01-0{day}").mkdir(parents=True)
        pd.DataFrame({"id": [day]}).to_parquet(
            root / f"event_date=2024-01-0{day}" / f"part{day}.parquet",
            index=False,
        )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(root),
        parquet_start_date=dt.date(2024, 1, 1),
        parquet_end_date=dt.date(2024, 1, 2),
        partition_on=["event_date"],
    )

    with ParquetDataResource(config) as resource:
        discovered = resource._discover_partitioned_files()

    filenames = {f.rsplit("/", 1)[-1] for f in discovered}
    assert filenames == {"part1.parquet", "part2.parquet"}


def test_parquet_partitioned_hive_discovery_falls_back_to_listing_for_explicit_pyarrow_fs(
    temp_project_root,
    monkeypatch,
) -> None:
    storage_path = "etl-dst/bronze/logistics/mobile/gps"
    root = temp_project_root / storage_path
    (root / "partition_date=2026-01-01").mkdir(parents=True)
    (root / "partition_date=2026-04-01").mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(
        root / "partition_date=2026-01-01" / "part1.parquet", index=False
    )
    pd.DataFrame({"id": [2]}).to_parquet(
        root / "partition_date=2026-04-01" / "part2.parquet", index=False
    )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=storage_path,
        parquet_start_date=dt.date(2026, 1, 1),
        parquet_end_date=dt.date(2026, 3, 31),
        partition_on=["partition_date"],
    )

    original_dataset = ds.dataset

    def dataset_with_failure(*args, **kwargs) -> ds.Dataset:
        if isinstance(kwargs.get("filesystem"), pafs.SubTreeFileSystem):
            raise OSError("simulated remote dataset discovery failure")
        return original_dataset(*args, **kwargs)

    monkeypatch.setattr(ds, "dataset", dataset_with_failure)
    import boti.core.secure_io as secure_io

    monkeypatch.setattr(secure_io, "is_secure_path", lambda target, allowed: True)

    fs = pafs.SubTreeFileSystem(str(temp_project_root), pafs.LocalFileSystem())
    with ParquetDataResource(config, fs=fs) as resource:
        discovered = resource._discover_partitioned_files()

    assert len(discovered) == 1
    assert discovered[0].endswith("part1.parquet")


def test_parquet_determine_recency_respects_max_age(temp_project_root) -> None:
    file_path = temp_project_root / "data" / "freshness.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(file_path, index=False)
    stale_timestamp = dt.datetime.now().timestamp() - (120 * 60)
    file_path.touch()
    Path(file_path).chmod(0o600)
    import os

    os.utime(file_path, (stale_timestamp, stale_timestamp))

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="freshness.parquet",
        parquet_max_age_minutes=5,
    )

    with ParquetDataResource(config) as resource:
        assert resource.determine_recency() is False


def test_parquet_determine_recency_uses_single_info_lookup(temp_project_root) -> None:
    class RecordingFS:
        protocol = "memory"

        def __init__(self) -> None:
            self.info_calls = 0
            self.modified_calls = 0

        def info(self, _path: str) -> dict[str, dt.datetime]:
            self.info_calls += 1
            return {"mtime": dt.datetime.now(dt.UTC)}

        def modified(self, _path: str) -> None:
            self.modified_calls += 1
            raise AssertionError("modified() should not run when info() already succeeded")

    file_path = temp_project_root / "data" / "single_info.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="single_info.parquet",
        parquet_max_age_minutes=5,
    )
    fs = RecordingFS()

    with ParquetDataResource(config, fs=fs) as resource:
        assert resource.determine_recency() is True

    assert fs.info_calls == 1
    assert fs.modified_calls == 0


def test_parquet_get_file_info_raises_on_non_missing_metadata_error(temp_project_root) -> None:
    class BrokenFS:
        protocol = "memory"

        def info(self, _path: str) -> None:
            raise PermissionError("permission denied")

    file_path = temp_project_root / "data" / "broken.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="broken.parquet",
        parquet_max_age_minutes=5,
    )

    with ParquetDataResource(config, fs=BrokenFS()) as resource:
        with pytest.raises(RuntimeError, match="Failed to read parquet file metadata"):
            resource._get_file_info(str(file_path))


def test_parquet_resource_blocks_local_paths_outside_project_root(
    temp_project_root, tmp_path
) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    pd.DataFrame({"id": [1]}).to_parquet(outside_dir / "secret.parquet", index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(outside_dir),
        parquet_filename="secret.parquet",
    )

    with ParquetDataResource(config) as resource:
        resource.allowed_paths = [resource.project_root]
        with pytest.raises(PermissionError, match="outside the configured sandbox roots"):
            resource.load_files()
