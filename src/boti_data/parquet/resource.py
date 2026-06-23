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
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from boti.core import (
    FilesystemConfig,
    ResourceConfig,
    SecureResource,
    create_filesystem,
    is_valid_identifier,
)
from fsspec.implementations.local import LocalFileSystem
from pydantic import Field, field_validator, model_validator

from boti_data.filters import FilterHandler


def create_local_filesystem() -> fsspec.AbstractFileSystem:
    """Create the default local filesystem for Parquet resources."""
    return fsspec.filesystem("file")


class ParquetDataConfig(ResourceConfig):
    """Validated configuration for Parquet-backed discovery and loading."""

    parquet_storage_path: str | None = Field(default=None)
    parquet_filename: str | None = Field(default=None)
    parquet_start_date: dt.date | None = Field(default=None)
    parquet_end_date: dt.date | None = Field(default=None)
    parquet_max_age_minutes: int = Field(default=0, ge=0)
    partition_on: list[str] | None = Field(default=None)
    filesystem_profile: str | None = Field(default=None)

    @field_validator("parquet_storage_path")
    @classmethod
    def validate_storage_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("parquet_storage_path must be specified.")
        return normalized.rstrip("/")

    @field_validator("parquet_filename")
    @classmethod
    def validate_filename(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("parquet_filename must be a simple file name, not a path.")
        return value

    @field_validator("partition_on")
    @classmethod
    def validate_partition_on(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("partition_on must contain at least one partition field.")
        if any(not is_valid_identifier(item) for item in value):
            raise ValueError("partition_on values must be valid identifiers.")
        return value

    @field_validator("filesystem_profile")
    @classmethod
    def validate_filesystem_profile(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("filesystem_profile must not be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_date_range(self) -> ParquetDataConfig:
        if self.parquet_storage_path is None and self.filesystem_profile is None:
            raise ValueError("Either parquet_storage_path or filesystem_profile must be provided.")
        if bool(self.parquet_start_date) != bool(self.parquet_end_date):
            raise ValueError(
                "Both parquet_start_date and parquet_end_date must be provided, or neither."
            )
        if (
            self.parquet_start_date is not None
            and self.parquet_end_date is not None
            and self.parquet_end_date < self.parquet_start_date
        ):
            raise ValueError("parquet_end_date cannot be before parquet_start_date.")
        return self


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
        self._filesystem_config: FilesystemConfig | None = None
        self._filesystem_adapter: Any | None = None
        resolved_fs_factory = fs_factory
        if fs is None and resolved_fs_factory is None and config.filesystem_profile is not None:
            if catalog is None:
                raise ValueError(
                    "ParquetDataResource requires a catalog when filesystem_profile is configured."
                )
            self._filesystem_config = catalog.filesystem_config(config.filesystem_profile)
            # Route profile-backed filesystem creation through the catalog adapter
            # so adapter retry/cache behavior is shared across callers.
            self._filesystem_adapter = catalog.filesystem_adapter(config.filesystem_profile)
            resolved_fs_factory = self._filesystem_adapter.get_filesystem

        super().__init__(
            config=config,
            fs=fs,
            fs_factory=resolved_fs_factory or (None if fs is not None else create_local_filesystem),
        )
        self.config = config
        # Eagerly validate the storage path at construction time for local paths
        # so path-traversal attempts are caught before any data access.
        if config.parquet_storage_path is not None:
            storage = config.parquet_storage_path
            parsed = __import__("urllib.parse", fromlist=["urlparse"]).urlparse(storage)
            if not parsed.scheme or parsed.scheme == "file":
                local = parsed.path if parsed.scheme == "file" else storage
                if "\x00" in local:
                    raise ValueError(
                        "parquet_storage_path contains a null byte and has been rejected."
                    )
                self.get_secure_path(local)

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        # Adapter instances can carry runtime-only locks and cache state.
        state.pop("_filesystem_adapter", None)
        return state

    def _restore_runtime_state(self) -> None:
        super()._restore_runtime_state()
        if not self._is_closed and self.fs is None and self._fs_factory is None:
            if self._filesystem_adapter is not None:
                self._fs_factory = self._filesystem_adapter.get_filesystem
            elif self._filesystem_config is not None:
                self._fs_factory = functools.partial(create_filesystem, self._filesystem_config)
            else:
                self._fs_factory = create_local_filesystem
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

    def _uses_nonlocal_explicit_fs(self) -> bool:
        fs = self.require_fs()
        if isinstance(fs, pafs.FileSystem):
            return not isinstance(fs, pafs.LocalFileSystem)
        if isinstance(fs, LocalFileSystem):
            return False
        protocol = getattr(fs, "protocol", None)
        if isinstance(protocol, tuple):
            return "file" not in protocol
        if isinstance(protocol, str):
            return protocol != "file"
        return True

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
            return self._empty_ddf()

        self.logger.debug(f"Loading {files_to_load} from {self.parquet_storage_path}")
        filesystem = self.require_fs()
        read_kwargs: dict[str, Any] = {
            "filesystem": filesystem,
            "filters": filters,
            "columns": columns,
            "ignore_metadata_file": True,
        }
        if isinstance(filesystem, pafs.FileSystem):
            pass
        else:
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
            [self._normalized_arrow_load_path(path) for path in files_to_load],
            filesystem=self._arrow_filesystem(),
            format="parquet",
        )
        expression = self._raw_filters_to_expression(filters)
        return dataset.to_table(filter=expression, columns=columns)

    def load_filtered(
        self,
        filters: dict[str, Any] | None = None,
        *,
        columns: list[str] | None = None,
    ) -> dd.DataFrame:
        """Load Parquet data from a high-level filter spec with pushdown + residual masking."""
        filter_handler = FilterHandler(backend="dask", logger=self.logger, debug=self.debug)
        pushdown_filters, residual_filters = filter_handler.split_pushdown_and_residual(
            filters or {}
        )
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
        pushdown_filters, residual_filters = filter_handler.split_pushdown_and_residual(
            filters or {}
        )
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

    def _resolve_files_to_load(self) -> list[str]:
        if self.parquet_filename:
            if not self.determine_recency():
                return []
            return [self._normalized_load_path(self.parquet_full_path)]
        if self.config.parquet_start_date and self.config.parquet_end_date:
            return self._discover_partitioned_files()

        self.logger.debug(
            f"No dates or filename provided. Scanning all files in {self.parquet_full_path}"
        )
        return self._discover_all_files()

    def _discover_partitioned_files(self) -> list[str]:
        source_path, filesystem = self._dataset_source()
        partitioning: ds.Partitioning | str
        partition_key: str

        if self.config.partition_on:
            partitioning = ds.partitioning(flavor="hive")
            partition_key = self.config.partition_on[0]
        else:
            partitioning = ds.partitioning(
                schema=pa.schema([("year", pa.int32()), ("month", pa.int32()), ("day", pa.int32())])
            )
            partition_key = "year"

        try:
            dataset = ds.dataset(
                source=source_path,
                filesystem=filesystem,
                format="parquet",
                partitioning=partitioning,
            )
        except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException) as exc:
            if self._is_missing_path_error(exc):
                self.logger.warning("Parquet path does not exist: %s", self.parquet_storage_path)
                return []
            return self._discover_partitioned_files_via_listing(partition_key)

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
        found_files = [self._restore_protocol(fragment.path) for fragment in fragments]
        if not found_files:
            return self._discover_partitioned_files_via_listing(partition_key)

        if self.debug:
            self.logger.debug(f"Requested range: {start_date} to {end_date}")
            self.logger.debug(f"Actual files found: {len(found_files)}")

        return found_files

    def _discover_partitioned_files_via_listing(self, partition_key: str) -> list[str]:
        start_date = self.config.parquet_start_date
        end_date = self.config.parquet_end_date
        assert start_date is not None and end_date is not None

        files: list[str] = []
        for path in self._list_candidate_files():
            if not path.endswith(".parquet"):
                continue
            if self._path_matches_partition_range(path, partition_key, start_date, end_date):
                files.append(self._restore_protocol(path))
        return sorted(files)

    @staticmethod
    def _is_missing_path_error(exc: Exception) -> bool:
        if isinstance(exc, FileNotFoundError):
            return True
        errno = getattr(exc, "errno", None)
        if errno in {2, 20}:
            return True
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "not found",
                "no such file",
                "does not exist",
                "path does not exist",
                "no such bucket",
                "bucket does not exist",
            )
        )

    @staticmethod
    def _filesystem_runtime_error(action: str, path: str, exc: Exception) -> RuntimeError:
        return RuntimeError(f"Failed to {action} for {path!r}: {exc}")

    def _list_candidate_files(self) -> list[str]:
        base_path = self.parquet_storage_path
        fs = self.require_fs()
        try:
            if isinstance(fs, pafs.FileSystem):
                selector = pafs.FileSelector(base_dir=base_path, recursive=True)
                infos = fs.get_file_info(selector)
                return [
                    info.path for info in infos if getattr(info, "type", None) == pafs.FileType.File
                ]

            if hasattr(fs, "find"):
                return list(fs.find(base_path))
            if hasattr(fs, "glob"):
                return [
                    path
                    for path in fs.glob(f"{base_path.rstrip('/')}/**")
                    if not path.endswith("/")
                ]
        except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException) as exc:
            if self._is_missing_path_error(exc):
                self.logger.warning("Parquet path does not exist for listing: %s", base_path)
                return []
            raise
        return []

    @staticmethod
    def _path_matches_partition_range(
        path: str,
        partition_key: str,
        start_date: dt.date,
        end_date: dt.date,
    ) -> bool:
        marker = f"{partition_key}="
        for segment in path.split("/"):
            if not segment.startswith(marker):
                continue
            partition_value = segment[len(marker) :]
            try:
                partition_date = dt.date.fromisoformat(partition_value)
            except ValueError:
                return start_date.isoformat() <= partition_value <= end_date.isoformat()
            return start_date <= partition_date <= end_date
        return False

    def _discover_all_files(self) -> list[str]:
        source_path, filesystem = self._dataset_source()
        try:
            dataset = ds.dataset(
                source_path,
                filesystem=filesystem,
                format="parquet",
                partitioning=None,
            )
        except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException) as exc:
            if self._is_missing_path_error(exc):
                self.logger.warning("Parquet path does not exist: %s", self.parquet_storage_path)
                return []
            raise self._filesystem_runtime_error(
                "scan parquet dataset", self.parquet_storage_path, exc
            ) from exc
        return [self._restore_protocol(path) for path in dataset.files]

    def determine_recency(self) -> bool:
        path = self._normalized_load_path(self.parquet_full_path)
        if not path.endswith(".parquet"):
            return True

        info = self._get_file_info(path)
        if info is None:
            return False

        modified_at = self._get_mtime_from_info(info)
        if self.config.parquet_max_age_minutes == 0:
            return modified_at is not None
        if modified_at is None:
            return False

        now = dt.datetime.now(dt.UTC)
        if modified_at.tzinfo is None:
            modified_at = modified_at.replace(tzinfo=dt.UTC)
        return (now - modified_at) <= dt.timedelta(minutes=self.config.parquet_max_age_minutes)

    def _get_file_info(self, path: str) -> dict[str, Any] | None:
        fs = self.require_fs()
        if isinstance(fs, pafs.FileSystem):
            try:
                info = fs.get_file_info(path)
            except Exception as exc:
                if self._is_missing_path_error(exc):
                    return None
                raise self._filesystem_runtime_error(
                    "read parquet file metadata", path, exc
                ) from exc
            if getattr(info, "type", None) == pafs.FileType.NotFound:
                return None
            payload: dict[str, Any] = {}
            size = getattr(info, "size", None)
            if isinstance(size, int):
                payload["size"] = size
            modified = getattr(info, "mtime", None)
            if modified is not None:
                payload["mtime"] = modified
            return payload
        try:
            return fs.info(path)
        except Exception as exc:
            if self._is_missing_path_error(exc):
                return None
            if hasattr(fs, "modified"):
                try:
                    modified = fs.modified(path)
                except Exception as modified_exc:
                    if self._is_missing_path_error(modified_exc):
                        return None
                    raise self._filesystem_runtime_error(
                        "read parquet file metadata",
                        path,
                        modified_exc,
                    ) from modified_exc
                if modified is not None:
                    return {"mtime": modified}
            raise self._filesystem_runtime_error("read parquet file metadata", path, exc) from exc

    @staticmethod
    def _get_mtime_from_info(info: dict[str, Any]) -> dt.datetime | None:
        modified = info.get("mtime")
        if isinstance(modified, dt.datetime):
            return modified
        if isinstance(modified, (int, float)):
            return dt.datetime.fromtimestamp(modified, tz=dt.UTC)
        if isinstance(modified, str):
            try:
                return dt.datetime.fromisoformat(modified)
            except ValueError:
                return None
        return None

    def _dataset_source(self) -> tuple[str, Any | None]:
        parsed = urlparse(self.parquet_storage_path)
        if parsed.scheme and parsed.scheme != "file":
            return self._strip_protocol(self.parquet_storage_path), self._arrow_filesystem()
        if self._uses_nonlocal_explicit_fs():
            return self.parquet_storage_path, self._arrow_filesystem()

        local_path = str(self._secure_local_path(self.parquet_storage_path))
        return local_path, None

    def _arrow_filesystem(self) -> pafs.FileSystem | None:
        parsed = urlparse(self.parquet_storage_path)
        if self._uses_nonlocal_explicit_fs():
            fs = self.require_fs()
            if isinstance(fs, pafs.FileSystem):
                return fs
            return pafs.PyFileSystem(pafs.FSSpecHandler(fs))
        if parsed.scheme and parsed.scheme != "file":
            return pafs.PyFileSystem(pafs.FSSpecHandler(self.require_fs()))
        return None

    def _normalized_load_path(self, path: str) -> str:
        parsed = urlparse(path)
        if self._uses_nonlocal_explicit_fs():
            return path
        if parsed.scheme and parsed.scheme != "file":
            return path
        return str(self._secure_local_path(path))

    def _normalized_arrow_load_path(self, path: str) -> str:
        parsed = urlparse(path)
        if self._uses_nonlocal_explicit_fs():
            return path
        if parsed.scheme and parsed.scheme != "file":
            return self._strip_protocol(path)
        return str(self._secure_local_path(path))

    def _secure_local_path(self, path: str) -> Path:
        parsed = urlparse(path)
        local_path = parsed.path if parsed.scheme == "file" else path
        # Guard against null-byte injection which can truncate paths on some systems.
        if "\x00" in local_path:
            raise ValueError(
                "Path contains a null byte and has been rejected for security reasons."
            )
        return self.get_secure_path(local_path)

    @staticmethod
    def _strip_protocol(path: str) -> str:
        return path.split("://", 1)[1] if "://" in path else path

    @staticmethod
    def _restore_protocol(path: str) -> str:
        return str(path)

    @staticmethod
    def _raw_filters_to_expression(filters: list[Any] | None) -> ds.Expression | None:
        if not filters:
            return None

        expression: ds.Expression | None = None
        for field, op, value in filters:
            clause: ds.Expression
            if op == "=":
                clause = ds.field(field) == value
            elif op == "!=":
                clause = ds.field(field) != value
            elif op == ">":
                clause = ds.field(field) > value
            elif op == ">=":
                clause = ds.field(field) >= value
            elif op == "<":
                clause = ds.field(field) < value
            elif op == "<=":
                clause = ds.field(field) <= value
            elif op == "in":
                clause = ds.field(field).isin(value)
            elif op == "not in":
                clause = ~ds.field(field).isin(value)
            else:
                raise ValueError(f"Unsupported parquet pushdown operator for Arrow load: {op!r}")
            expression = clause if expression is None else expression & clause
        return expression

    def _empty_ddf(self) -> dd.DataFrame:
        try:
            fs = self.require_fs()
            storage_path = self.parquet_storage_path
            for meta_file in ("_common_metadata", "_metadata"):
                meta_path = posixpath.join(storage_path, meta_file)
                try:
                    if isinstance(fs, pafs.FileSystem):
                        if fs.get_file_info(meta_path).type != pafs.FileType.File:
                            continue
                    elif not fs.exists(meta_path):
                        continue
                    schema = pq.read_schema(meta_path, filesystem=fs)
                    return dd.from_pandas(
                        pd.DataFrame(columns=schema.names), npartitions=1
                    )
                except Exception:
                    continue
        except Exception:
            pass
        return dd.from_pandas(pd.DataFrame(), npartitions=1)

    @staticmethod
    def ensure_file_extension(filepath: str, extension: str) -> str:
        return filepath if filepath.endswith(f".{extension}") else f"{filepath}.{extension}"
