"""
Tests for the Parquet data resource.
"""

from __future__ import annotations

import datetime as dt
import pickle
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pytest
from boti.core.filesystem import FilesystemConfig
from pydantic import ValidationError

from boti_data import ConnectionCatalog, DataHelper, ParquetDataConfig, ParquetDataResource


class StubLogger:
    def debug(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def error(self, *_args, **_kwargs) -> None:
        pass


def test_parquet_config_validates_date_range_and_partition_fields():
    with pytest.raises(ValueError, match="Both parquet_start_date and parquet_end_date"):
        ParquetDataConfig(parquet_storage_path="/tmp/data", parquet_start_date=dt.date(2024, 1, 1))

    with pytest.raises(ValueError, match="cannot be before"):
        ParquetDataConfig(
            parquet_storage_path="/tmp/data",
            parquet_start_date=dt.date(2024, 1, 2),
            parquet_end_date=dt.date(2024, 1, 1),
        )

    with pytest.raises(ValueError, match="simple file name"):
        ParquetDataConfig(parquet_storage_path="/tmp/data", parquet_filename="../escape.parquet")

    with pytest.raises(ValueError, match="valid identifiers"):
        ParquetDataConfig(parquet_storage_path="/tmp/data", partition_on=["bad-field"])


def test_parquet_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ParquetDataConfig(parquet_storage_path="/tmp/data", unexpected=True)


def test_filesystem_config_supports_named_env_prefixes(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "WAREHOUSE_FS_TYPE=file",
                f"WAREHOUSE_FS_PATH={tmp_path}",
                "WAREHOUSE_FS_VERIFY_SSL=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = FilesystemConfig.from_env_prefix("WAREHOUSE_", env_file=env_file)

    assert config.fs_type == "file"
    assert Path(config.fs_path) == tmp_path
    assert config.fs_verify_ssl is False


def test_parquet_filesystem_profile_uses_catalog_adapter_fs_factory(temp_project_root):
    root = temp_project_root / "profile_data"
    root.mkdir(parents=True)
    pd.DataFrame({"id": [1], "name": ["Alice"]}).to_parquet(root / "users.parquet", index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        parquet_filename="users",
        filesystem_profile="warehouse",
    )

    fs = fsspec.filesystem("file")

    class FakeAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def get_filesystem(self):
            self.calls += 1
            return fs

    adapter = FakeAdapter()

    class FakeCatalog:
        def filesystem_config(self, name: str) -> FilesystemConfig:
            assert name == "warehouse"
            return FilesystemConfig(fs_type="file", fs_path=str(root))

        def filesystem_adapter(self, name: str):
            assert name == "warehouse"
            return adapter

    with ParquetDataResource(config, catalog=FakeCatalog()) as resource:
        loaded = resource.load_files().compute().sort_values("id").reset_index(drop=True)

    assert loaded["name"].tolist() == ["Alice"]
    assert adapter.calls == 1


def test_parquet_single_file_loading(temp_project_root):
    file_path = temp_project_root / "data" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    expected = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    expected.to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="users",
        allow_pickle=True,
    )

    with ParquetDataResource(config) as resource:
        loaded = resource.load_files().compute().sort_values("id").reset_index(drop=True)

    pd.testing.assert_frame_equal(loaded[["id", "name"]], expected, check_dtype=False)


def test_parquet_load_filtered_combines_pushdown_and_residual(temp_project_root):
    file_path = temp_project_root / "data" / "tickets.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "status": ["active", "active", "inactive"],
            "description": ["urgent issue", "routine followup", "urgent backlog"],
        }
    ).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="tickets",
    )

    with ParquetDataResource(config) as resource:
        loaded = resource.load_filtered(
            {"status__exact": "active", "description__icontains": "urgent"}
        ).compute().sort_values("id").reset_index(drop=True)

    assert loaded["id"].tolist() == [1]


def test_parquet_load_files_raw_filters_still_work(temp_project_root):
    file_path = temp_project_root / "data" / "events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="events",
    )

    with ParquetDataResource(config) as resource:
        loaded = resource.load_files(filters=[("status", "=", "inactive")]).compute()

    assert loaded["id"].tolist() == [2]


def test_datahelper_parquet_applies_bare_kwarg_filters(temp_project_root):
    """Regression: ``DataHelper(backend='parquet')`` must apply bare-kwarg filters.

    The parquet strategy previously built its request from the typed
    ``GatewayLoadRequest`` (which forbids extra fields), so bare runtime filter
    kwargs never reached ``request.filters`` and the full dataset was returned
    unfiltered. Only ``ParquetReader`` worked, because it pre-folds bare kwargs
    into an explicit ``filters=`` mapping.
    """
    file_path = temp_project_root / "data" / "gps.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "associate_id": [27, 2285, 2285, 4499],
        }
    ).to_parquet(file_path, index=False)

    helper = DataHelper(
        backend="parquet",
        storage_path=str(file_path.parent),
        parquet_filename="gps",
    )
    try:
        # dask structured path (load_filtered)
        exact = helper.load(associate_id=2285, return_type="dask").compute()
        assert sorted(exact["id"].tolist()) == [2, 3]

        # arrow/pandas structured path (load_filtered_arrow)
        subset = helper.load(associate_id__in=[27, 4499], return_type="pandas")
        assert sorted(subset["id"].tolist()) == [1, 4]

        gte = helper.load(associate_id__gte=2285, return_type="pandas")
        assert sorted(gte["id"].tolist()) == [2, 3, 4]
    finally:
        helper.close()


def test_parquet_coerces_temporal_filter_on_string_column(temp_project_root):
    """Regression: date/datetime filter values are coerced to ISO strings for
    string-typed columns instead of raising ``ArrowNotImplementedError``.

    A ``date`` value compared against a text column (an ISO date stored as a
    string) has no PyArrow comparison kernel on either the pushdown or residual
    path; coercing it to its ISO string turns the comparison into a correct
    lexicographic one.
    """
    file_path = temp_project_root / "data" / "orders.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "order_date": ["2026-04-15", "2026-04-17", "2026-04-25"],
        }
    ).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="orders",
    )

    with ParquetDataResource(config) as resource:
        # dask pushdown + residual path
        loaded = (
            resource.load_filtered({"order_date__gte": dt.date(2026, 4, 17)})
            .compute()
            .sort_values("id")
        )
        assert loaded["id"].tolist() == [2, 3]

        # arrow/pandas path
        table = resource.load_filtered_arrow({"order_date__lt": dt.date(2026, 4, 17)})
        assert table.column("id").to_pylist() == [1]


def test_datahelper_parquet_load_period_on_string_date_column(temp_project_root):
    """Regression: ``load_period`` over a string date column filters correctly.

    Exercises both fixes together through the public API — the same path the
    ``HybridDataset`` parquet branch uses. ``load_period`` emits ``date`` bounds
    as bare ``__gte``/``__lte`` kwargs, which must both reach the parquet filter
    and coerce against the string ``order_date`` column.
    """
    file_path = temp_project_root / "data" / "orders.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "order_date": ["2026-04-15", "2026-04-17", "2026-04-25"],
        }
    ).to_parquet(file_path, index=False)

    helper = DataHelper(
        backend="parquet",
        storage_path=str(file_path.parent),
        parquet_filename="orders",
    )
    try:
        window = helper.load_period(
            "order_date", "2026-04-15", "2026-04-19", return_type="pandas"
        )
    finally:
        helper.close()

    assert sorted(window["id"].tolist()) == [1, 2]


def test_parquet_discover_all_files(temp_project_root):
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


def test_parquet_discover_all_files_raises_on_non_missing_scan_error(temp_project_root, monkeypatch):
    root = temp_project_root / "scan_failure"
    root.mkdir(parents=True)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(root),
    )

    def dataset_with_permission_error(*_args, **_kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(ds, "dataset", dataset_with_permission_error)

    with ParquetDataResource(config) as resource:
        with pytest.raises(RuntimeError, match="Failed to scan parquet dataset"):
            resource._discover_all_files()


def test_parquet_partitioned_directory_discovery(temp_project_root):
    root = temp_project_root / "partitioned"
    (root / "2024" / "1" / "1").mkdir(parents=True)
    (root / "2025" / "1" / "1").mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(root / "2024" / "1" / "1" / "part_2024.parquet", index=False)
    pd.DataFrame({"id": [2]}).to_parquet(root / "2025" / "1" / "1" / "part_2025.parquet", index=False)

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


def test_parquet_partitioned_hive_discovery(temp_project_root):
    root = temp_project_root / "hive_partitioned"
    (root / "event_date=2024-01-01").mkdir(parents=True)
    (root / "event_date=2024-01-02").mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(root / "event_date=2024-01-01" / "part1.parquet", index=False)
    pd.DataFrame({"id": [2]}).to_parquet(root / "event_date=2024-01-02" / "part2.parquet", index=False)

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


def test_parquet_partitioned_hive_negative_pruning(temp_project_root):
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

    assert len(discovered) == 2
    assert all(f.endswith("parquet") for f in discovered)
    assert any(f.endswith("part1.parquet") for f in discovered)
    assert any(f.endswith("part2.parquet") for f in discovered)
    assert not any(f.endswith("part3.parquet") for f in discovered)


def test_parquet_partitioned_hive_discovery_falls_back_to_listing_for_explicit_pyarrow_fs(
    temp_project_root,
    monkeypatch,
):
    storage_path = "etl-dst/bronze/logistics/mobile/gps"
    root = temp_project_root / storage_path
    (root / "partition_date=2026-01-01").mkdir(parents=True)
    (root / "partition_date=2026-04-01").mkdir(parents=True)
    pd.DataFrame({"id": [1]}).to_parquet(root / "partition_date=2026-01-01" / "part1.parquet", index=False)
    pd.DataFrame({"id": [2]}).to_parquet(root / "partition_date=2026-04-01" / "part2.parquet", index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=storage_path,
        parquet_start_date=dt.date(2026, 1, 1),
        parquet_end_date=dt.date(2026, 3, 31),
        partition_on=["partition_date"],
    )

    original_dataset = ds.dataset

    def dataset_with_failure(*args, **kwargs):
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


def test_parquet_determine_recency_respects_max_age(temp_project_root):
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


def test_parquet_determine_recency_uses_single_info_lookup(temp_project_root):
    class RecordingFS:
        protocol = "memory"

        def __init__(self) -> None:
            self.info_calls = 0
            self.modified_calls = 0

        def info(self, _path: str):
            self.info_calls += 1
            return {"mtime": dt.datetime.now(dt.UTC)}

        def modified(self, _path: str):
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


def test_parquet_get_file_info_raises_on_non_missing_metadata_error(temp_project_root):
    class BrokenFS:
        protocol = "memory"

        def info(self, _path: str):
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


def test_parquet_resource_blocks_local_paths_outside_project_root(temp_project_root, tmp_path):
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


def test_parquet_resource_uses_named_filesystem_profile(temp_project_root):
    file_path = temp_project_root / "profiled" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    expected = pd.DataFrame({"id": [1], "name": ["Alice"]})
    expected.to_parquet(file_path, index=False)

    catalog = ConnectionCatalog()
    catalog.register_filesystem(
        "warehouse",
        FilesystemConfig(fs_type="file", fs_path=str(file_path.parent)),
    )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        filesystem_profile="warehouse",
    )

    with ParquetDataResource(config, catalog=catalog) as resource:
        loaded = resource.load_files().compute().sort_values("id").reset_index(drop=True)

    pd.testing.assert_frame_equal(loaded[["id", "name"]], expected, check_dtype=False)


def test_parquet_resource_is_pickleable_for_distributed_use(temp_project_root):
    file_path = temp_project_root / "distributed" / "users.parquet"
    file_path.parent.mkdir(parents=True)
    expected = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    expected.to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="users",
        allow_pickle=True,
    )

    resource = ParquetDataResource(config)
    restored = None
    try:
        with ParquetDataResource.trusted_unpickle_scope():
            restored = pickle.loads(pickle.dumps(resource))
        loaded = restored.load_files().compute().sort_values("id").reset_index(drop=True)
    finally:
        resource.close()
        if restored is not None:
            restored.close()

    pd.testing.assert_frame_equal(loaded[["id", "name"]], expected, check_dtype=False)
