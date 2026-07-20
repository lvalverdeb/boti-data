"""
Parquet-backed data loading resources.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import functools
import posixpath
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import dask.dataframe as dd
import fsspec
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from boti.core.filesystem import create_filesystem
from boti.core.secure_io import SecureResource

from boti_data.filters import FilterHandler

from . import discovery, filesystem, schema_filters
from .config import ParquetDataConfig

__all__ = ["ParquetDataConfig", "ParquetDataResource"]


class ParquetDataResource(SecureResource):
    """Secure resource for discovering and loading Parquet data."""

    def __init__(
        self,
        config: ParquetDataConfig,
        *,
        fs: fsspec.AbstractFileSystem | None = None,
        fs_factory: Any | None = None,
        catalog: Any | None = None,
    ) -> None:
        self._filesystem_config, resolved_fs_factory = filesystem.resolve_filesystem_factory(
            config,
            fs=fs,
            fs_factory=fs_factory,
            catalog=catalog,
        )

        super().__init__(
            config=config,
            fs=fs,
            fs_factory=resolved_fs_factory
            or (None if fs is not None else filesystem.create_local_filesystem),
        )
        self.config = config
        if self.config.parquet_storage_path is not None:
            self._secure_local_path(self.config.parquet_storage_path)

    def _restore_runtime_state(self) -> None:
        super()._restore_runtime_state()
        if not self._is_closed and self.fs is None and self._fs_factory is None:
            if self._filesystem_config is not None:
                self._fs_factory = functools.partial(create_filesystem, self._filesystem_config)
            else:
                self._fs_factory = filesystem.create_local_filesystem
            self._owns_fs = True

    @property
    def parquet_storage_path(self) -> str:
        if self.config.parquet_storage_path is not None:
            return self.config.parquet_storage_path
        if self._filesystem_config is not None:
            return self._filesystem_config.storage_path
        raise RuntimeError("Parquet storage path is not configured.")

    @property
    def parquet_filename(self) -> str | None:
        return self.config.parquet_filename

    @property
    def parquet_full_path(self) -> str:
        if self.parquet_filename:
            raw_path = posixpath.join(self.parquet_storage_path, self.parquet_filename)
            return self.ensure_file_extension(raw_path, "parquet")
        return self.parquet_storage_path

    def load_files(
        self,
        filters: list[Any] | None = None,
        *,
        columns: list[str] | None = None,
    ) -> dd.DataFrame:
        """Load Parquet data using the configured discovery strategy."""
        files_to_load = self._resolve_files_to_load()
        if not files_to_load:
            self.logger.warning("No files found.")
            return schema_filters.empty_ddf()

        self.logger.debug("Loading %s from %s", files_to_load, self.parquet_storage_path)
        fs_ = self.require_fs()
        read_kwargs: dict[str, Any] = {
            "filesystem": fs_,
            "filters": filters,
            "columns": columns,
            "ignore_metadata_file": True,
        }
        if not isinstance(fs_, pafs.FileSystem):
            read_kwargs["engine"] = "pyarrow"
            read_kwargs["aggregate_files"] = True

        return dd.read_parquet(
            files_to_load,
            **read_kwargs,
        )

    def load_arrow(
        self,
        filters: list[Any] | None = None,
        *,
        columns: list[str] | None = None,
    ) -> pa.Table:
        """Load Parquet data as a native Arrow table."""
        files_to_load = self._resolve_files_to_load()
        if not files_to_load:
            self.logger.warning("No files found.")
            return pa.table({})

        dataset = ds.dataset(
            [filesystem.normalized_arrow_load_path(self, path) for path in files_to_load],
            filesystem=filesystem.arrow_filesystem(self),
            format="parquet",
        )
        expression = schema_filters.raw_filters_to_expression(filters)
        return dataset.to_table(filter=expression, columns=columns)

    def load_filtered(
        self,
        filters: dict[str, Any] | None = None,
        *,
        columns: list[str] | None = None,
    ) -> dd.DataFrame:
        """Load Parquet data from a high-level filter spec with pushdown + residual masking."""
        filter_handler = FilterHandler(backend="dask", logger=self.logger, debug=self.debug)
        coerced = schema_filters.coerce_temporal_filters(self, filters or {})
        pushdown_filters, residual_filters = filter_handler.split_pushdown_and_residual(coerced)
        dataframe = self.load_files(filters=pushdown_filters or None, columns=columns)
        if residual_filters:
            dataframe = filter_handler.apply_filters(dataframe, filters=residual_filters)
        return dataframe

    def load_filtered_arrow(
        self,
        filters: dict[str, Any] | None = None,
        *,
        columns: list[str] | None = None,
    ) -> pa.Table:
        """Load Parquet data as Arrow with pushdown and Arrow residual filters."""
        filter_handler = FilterHandler(backend="arrow", logger=self.logger, debug=self.debug)
        coerced = schema_filters.coerce_temporal_filters(self, filters or {})
        pushdown_filters, residual_filters = filter_handler.split_pushdown_and_residual(coerced)
        table = self.load_arrow(filters=pushdown_filters or None, columns=columns)
        if residual_filters:
            table = filter_handler.apply_filters(table, filters=residual_filters)
        return table

    async def aload_files(
        self,
        filters: list[Any] | None = None,
        *,
        columns: list[str] | None = None,
    ) -> dd.DataFrame:
        """Async wrapper for parquet loading."""
        return await asyncio.to_thread(self.load_files, filters, columns=columns)

    async def aload_filtered(
        self,
        filters: dict[str, Any] | None = None,
        *,
        columns: list[str] | None = None,
    ) -> dd.DataFrame:
        """Async wrapper for high-level parquet filtering."""
        return await asyncio.to_thread(self.load_filtered, filters, columns=columns)

    def _resolve_filename_path(self) -> list[str]:
        if not self.determine_recency():
            return []
        return [filesystem.normalized_load_path(self, self.parquet_full_path)]

    def _resolve_files_to_load(self) -> list[str]:
        if self.parquet_filename:
            return self._resolve_filename_path()
        if self.config.parquet_start_date and self.config.parquet_end_date:
            return self._discover_partitioned_files()

        self.logger.debug(
            f"No dates or filename provided. Scanning all files in {self.parquet_full_path}"
        )
        return self._discover_all_files()

    def _partitioning_for_config(self) -> tuple[ds.Partitioning | str, str]:
        if self.config.partition_on:
            return ds.partitioning(flavor="hive"), self.config.partition_on[0]
        return (
            ds.partitioning(
                schema=pa.schema([("year", pa.int32()), ("month", pa.int32()), ("day", pa.int32())])
            ),
            "year",
        )

    def _open_partitioned_dataset(
        self,
        source_path: str,
        fs_: Any,
        partitioning: ds.Partitioning | str,
        partition_key: str,
    ) -> ds.Dataset | list[str]:
        """Returns the opened dataset, or a listing-based file list on failure/missing-path."""
        try:
            return ds.dataset(
                source=source_path,
                filesystem=fs_,
                format="parquet",
                partitioning=partitioning,
            )
        except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException) as exc:
            if discovery.is_missing_path_error(exc):
                self.logger.warning("Parquet path does not exist: %s", self.parquet_storage_path)
                return []
            return discovery.discover_partitioned_files_via_listing(self, partition_key)

    def _discover_partitioned_files(self) -> list[str]:
        source_path, fs_ = filesystem.dataset_source(self)
        partitioning, partition_key = self._partitioning_for_config()

        dataset = self._open_partitioned_dataset(source_path, fs_, partitioning, partition_key)
        if isinstance(dataset, list):
            return dataset

        start_date = self.config.parquet_start_date
        end_date = self.config.parquet_end_date
        assert start_date is not None and end_date is not None

        if self.config.partition_on:
            expression = (ds.field(partition_key) >= str(start_date)) & (
                ds.field(partition_key) <= str(end_date)
            )
        else:
            expression = (ds.field("year") >= start_date.year) & (ds.field("year") <= end_date.year)

        fragments = dataset.get_fragments(expression)
        found_files = [filesystem.restore_protocol(fragment.path) for fragment in fragments]
        if not found_files:
            return discovery.discover_partitioned_files_via_listing(self, partition_key)

        if self.debug:
            self.logger.debug("Requested range: %s to %s", start_date, end_date)
            self.logger.debug("Actual files found: %s", len(found_files))

        return found_files

    def _discover_all_files(self) -> list[str]:
        source_path, fs_ = filesystem.dataset_source(self)
        try:
            dataset = ds.dataset(
                source_path,
                filesystem=fs_,
                format="parquet",
                partitioning=None,
            )
        except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException) as exc:
            if discovery.is_missing_path_error(exc):
                self.logger.warning("Parquet path does not exist: %s", self.parquet_storage_path)
                return []
            raise discovery.filesystem_runtime_error(
                "scan parquet dataset", self.parquet_storage_path, exc
            ) from exc
        return [filesystem.restore_protocol(path) for path in dataset.files]

    def _is_recent_enough(self, modified_at: dt.datetime | None) -> bool:
        if self.config.parquet_max_age_minutes == 0:
            return modified_at is not None
        if modified_at is None:
            return False

        now = dt.datetime.now(dt.UTC)
        if modified_at.tzinfo is None:
            modified_at = modified_at.replace(tzinfo=dt.UTC)
        return (now - modified_at) <= dt.timedelta(minutes=self.config.parquet_max_age_minutes)

    def determine_recency(self) -> bool:
        path = filesystem.normalized_load_path(self, self.parquet_full_path)
        if not path.endswith(".parquet"):
            return True

        info = self._get_file_info(path)
        if info is None:
            return False

        modified_at = discovery.get_mtime_from_info(info)
        return self._is_recent_enough(modified_at)

    def _sum_file_sizes(self, files_to_load: list[str]) -> int | None:
        """Returns total byte size across files, or None if any file's size is unavailable."""
        total_bytes = 0
        for path in files_to_load:
            info = self._get_file_info(path)
            if info is None:
                return None
            size = info.get("size")
            if not isinstance(size, int):
                return None
            total_bytes += size
        return total_bytes

    def scan_summary(self, *, max_files: int) -> tuple[int | None, int | None]:
        """Cheap size probe for auto return-type decisions.

        Returns ``(row_count, total_bytes)``; either may be ``None`` when it
        cannot be determined cheaply (row_count is always unknown for
        Parquet without opening the files). Returns ``(None, None)`` when the
        file count exceeds ``max_files`` or file info is unavailable.
        """
        files_to_load = self._resolve_files_to_load()
        if not files_to_load:
            return 0, 0
        if len(files_to_load) > max_files:
            return None, None
        return None, self._sum_file_sizes(files_to_load)

    def _get_file_info(self, path: str) -> dict[str, Any] | None:
        fs_ = self.require_fs()
        return discovery.file_info_provider(self, fs_).info(path)

    def _secure_local_path(self, path: str) -> Path:
        parsed = urlparse(path)
        local_path = parsed.path if parsed.scheme == "file" else path
        return self.get_secure_path(local_path)

    @staticmethod
    def ensure_file_extension(filepath: str, extension: str) -> str:
        return filepath if filepath.endswith(f".{extension}") else f"{filepath}.{extension}"
