"""File discovery helpers for ParquetDataResource: recency, listing, and info lookups."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.fs as pafs

from .file_info import ArrowFileInfoProvider, FsspecFileInfoProvider
from .filesystem import restore_protocol

if TYPE_CHECKING:
    from .resource import ParquetDataResource


def is_missing_path_error(exc: Exception) -> bool:
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


def filesystem_runtime_error(action: str, path: str, exc: Exception) -> RuntimeError:
    return RuntimeError(f"Failed to {action} for {path!r}: {exc}")


def _list_via_arrow_filesystem(fs: pafs.FileSystem, base_path: str) -> list[str]:
    selector = pafs.FileSelector(base_dir=base_path, recursive=True)
    infos = fs.get_file_info(selector)
    return [info.path for info in infos if getattr(info, "type", None) == pafs.FileType.File]


def _list_via_filesystem(fs: Any, base_path: str) -> list[str] | None:
    """Returns the file listing for whichever filesystem API `fs` supports, or None."""
    if isinstance(fs, pafs.FileSystem):
        return _list_via_arrow_filesystem(fs, base_path)
    if hasattr(fs, "find"):
        return list(fs.find(base_path))
    if hasattr(fs, "glob"):
        return [path for path in fs.glob(f"{base_path.rstrip('/')}/**") if not path.endswith("/")]


def list_candidate_files(resource: ParquetDataResource) -> list[str]:
    base_path = resource.parquet_storage_path
    fs = resource.require_fs()
    try:
        result = _list_via_filesystem(fs, base_path)
    except (FileNotFoundError, OSError, pa.ArrowInvalid, pa.ArrowException) as exc:
        if is_missing_path_error(exc):
            resource.logger.warning("Parquet path does not exist for listing: %s", base_path)
            return []
        raise
    return result if result is not None else []


def path_matches_partition_range(
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


def discover_partitioned_files_via_listing(
    resource: ParquetDataResource, partition_key: str
) -> list[str]:
    start_date = resource.config.parquet_start_date
    end_date = resource.config.parquet_end_date
    assert start_date is not None and end_date is not None

    files: list[str] = []
    for path in list_candidate_files(resource):
        if not path.endswith(".parquet"):
            continue
        if path_matches_partition_range(path, partition_key, start_date, end_date):
            files.append(restore_protocol(path))
    return sorted(files)


def get_arrow_file_info(
    resource: ParquetDataResource, fs: pafs.FileSystem, path: str
) -> dict[str, Any] | None:
    return arrow_file_info_provider(resource, fs).info(path)


def get_fsspec_file_info(
    resource: ParquetDataResource, fs: Any, path: str
) -> dict[str, Any] | None:
    return fsspec_file_info_provider(resource, fs).info(path)


def try_fsspec_modified(
    resource: ParquetDataResource, fs: Any, path: str, original_exc: Exception
) -> dict[str, Any] | None:
    return fsspec_file_info_provider(resource, fs)._try_modified(path, original_exc)


def file_info_provider(
    resource: ParquetDataResource, fs: Any
) -> ArrowFileInfoProvider | FsspecFileInfoProvider:
    if isinstance(fs, pafs.FileSystem):
        return arrow_file_info_provider(resource, fs)
    return fsspec_file_info_provider(resource, fs)


def arrow_file_info_provider(
    resource: ParquetDataResource, fs: pafs.FileSystem
) -> ArrowFileInfoProvider:
    return ArrowFileInfoProvider(
        fs,
        is_missing_path_error=is_missing_path_error,
        filesystem_runtime_error=filesystem_runtime_error,
    )


def fsspec_file_info_provider(resource: ParquetDataResource, fs: Any) -> FsspecFileInfoProvider:
    return FsspecFileInfoProvider(
        fs,
        is_missing_path_error=is_missing_path_error,
        filesystem_runtime_error=filesystem_runtime_error,
    )


def _parse_iso_mtime(value: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def get_mtime_from_info(info: dict[str, Any]) -> dt.datetime | None:
    modified = info.get("mtime")
    if isinstance(modified, dt.datetime):
        return modified
    if isinstance(modified, (int, float)):
        return dt.datetime.fromtimestamp(modified, tz=dt.UTC)
    return _parse_iso_mtime(modified) if isinstance(modified, str) else None
