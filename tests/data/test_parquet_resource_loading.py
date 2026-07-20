"""
Parquet resource tests: config validation, filesystem setup, and file/filter
loading behavior.

Split out of test_parquet_resource.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import datetime as dt
import pickle
from pathlib import Path

import fsspec
import pandas as pd
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


def test_parquet_config_validates_date_range_and_partition_fields() -> None:
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


def test_parquet_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ParquetDataConfig(parquet_storage_path="/tmp/data", unexpected=True)


def test_filesystem_config_supports_named_env_prefixes(tmp_path) -> None:
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


def test_parquet_filesystem_profile_uses_catalog_adapter_fs_factory(temp_project_root) -> None:
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

        def get_filesystem(self) -> fsspec.AbstractFileSystem:
            self.calls += 1
            return fs

    adapter = FakeAdapter()

    class FakeCatalog:
        def filesystem_config(self, name: str) -> FilesystemConfig:
            assert name == "warehouse"
            return FilesystemConfig(fs_type="file", fs_path=str(root))

        def filesystem_adapter(self, name: str) -> FakeAdapter:
            assert name == "warehouse"
            return adapter

    with ParquetDataResource(config, catalog=FakeCatalog()) as resource:
        loaded_computed = resource.load_files().compute()
        loaded = loaded_computed.sort_values("id").reset_index(drop=True)

    assert loaded["name"].tolist() == ["Alice"]
    assert adapter.calls == 1


def test_parquet_single_file_loading(temp_project_root) -> None:
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
        loaded_computed = resource.load_files().compute()
        loaded = loaded_computed.sort_values("id").reset_index(drop=True)

    pd.testing.assert_frame_equal(loaded[["id", "name"]], expected, check_dtype=False)


def test_parquet_load_filtered_combines_pushdown_and_residual(temp_project_root) -> None:
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
        filtered = resource.load_filtered(
            {"status__exact": "active", "description__icontains": "urgent"}
        )
        loaded = filtered.compute().sort_values("id").reset_index(drop=True)

    assert loaded["id"].tolist() == [1]


def test_parquet_load_files_raw_filters_still_work(temp_project_root) -> None:
    file_path = temp_project_root / "data" / "events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2], "status": ["active", "inactive"]}).to_parquet(
        file_path, index=False
    )

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="events",
    )

    with ParquetDataResource(config) as resource:
        loaded = resource.load_files(filters=[("status", "=", "inactive")]).compute()

    assert loaded["id"].tolist() == [2]


def test_datahelper_parquet_applies_bare_kwarg_filters(temp_project_root) -> None:
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


def test_parquet_coerces_temporal_filter_on_string_column(temp_project_root) -> None:
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


def test_datahelper_parquet_load_period_on_string_date_column(temp_project_root) -> None:
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
        window = helper.load_period("order_date", "2026-04-15", "2026-04-19", return_type="pandas")
    finally:
        helper.close()

    assert sorted(window["id"].tolist()) == [1, 2]


def test_parquet_resource_uses_named_filesystem_profile(temp_project_root) -> None:
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
        loaded_computed = resource.load_files().compute()
        loaded = loaded_computed.sort_values("id").reset_index(drop=True)

    pd.testing.assert_frame_equal(loaded[["id", "name"]], expected, check_dtype=False)


def test_parquet_resource_is_pickleable_for_distributed_use(temp_project_root) -> None:
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
        restored_computed = restored.load_files().compute()
        loaded = restored_computed.sort_values("id").reset_index(drop=True)
    finally:
        resource.close()
        if restored is not None:
            restored.close()

    pd.testing.assert_frame_equal(loaded[["id", "name"]], expected, check_dtype=False)
