"""Filesystem resolution and path normalization for ParquetDataResource."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import fsspec
import pyarrow.fs as pafs
from boti.core.filesystem import FilesystemConfig, create_filesystem
from fsspec.implementations.local import LocalFileSystem

if TYPE_CHECKING:
    from .config import ParquetDataConfig
    from .resource import ParquetDataResource


def create_local_filesystem() -> fsspec.AbstractFileSystem:
    """Create the default local filesystem for Parquet resources."""
    return fsspec.filesystem("file")


def resolve_filesystem_factory(
    config: ParquetDataConfig,
    *,
    fs: fsspec.AbstractFileSystem | None,
    fs_factory: Any | None,
    catalog: Any | None,
) -> tuple[FilesystemConfig | None, Any | None]:
    if fs is not None or fs_factory is not None or config.filesystem_profile is None:
        return None, fs_factory
    if catalog is None:
        raise ValueError(
            "ParquetDataResource requires a catalog when filesystem_profile is configured."
        )
    filesystem_config = catalog.filesystem_config(config.filesystem_profile)
    adapter_factory = catalog_filesystem_factory(catalog, config.filesystem_profile)
    return filesystem_config, adapter_factory or functools.partial(
        create_filesystem,
        filesystem_config,
    )


def catalog_filesystem_factory(catalog: Any, profile: str) -> Any | None:
    if not hasattr(catalog, "filesystem_adapter"):
        return None
    adapter = catalog.filesystem_adapter(profile)
    if adapter is not None and hasattr(adapter, "get_filesystem"):
        return adapter.get_filesystem
    return None


def _protocol_is_nonlocal(protocol: Any) -> bool:
    if isinstance(protocol, tuple):
        return "file" not in protocol
    if isinstance(protocol, str):
        return protocol != "file"
    return True


def uses_nonlocal_explicit_fs(resource: ParquetDataResource) -> bool:
    fs = resource.require_fs()
    if isinstance(fs, pafs.FileSystem):
        return not isinstance(fs, pafs.LocalFileSystem)
    if isinstance(fs, LocalFileSystem):
        return False
    return _protocol_is_nonlocal(getattr(fs, "protocol", None))


def secure_local_path(resource: ParquetDataResource, path: str) -> str:
    """*path* confined to the sandbox, for a resource whose filesystem is local.

    A scheme other than ``file`` names a remote location, and a local
    filesystem cannot be the right one for it; letting it through would
    strip the scheme and read the remainder as a local path outside the
    sandbox's checks.
    """
    parsed = urlparse(path)
    if parsed.scheme and parsed.scheme != "file":
        raise PermissionError(
            f"{path!r} names a {parsed.scheme}:// location but this resource's filesystem "
            "is local. Configure a remote filesystem (filesystem_config or fs) for it."
        )
    return str(resource._secure_local_path(path))


def dataset_source(resource: ParquetDataResource) -> tuple[str, Any | None]:
    if uses_nonlocal_explicit_fs(resource):
        return strip_protocol(resource.parquet_storage_path), arrow_filesystem(resource)
    return secure_local_path(resource, resource.parquet_storage_path), None


def _as_arrow_filesystem(fs: Any) -> pafs.FileSystem:
    if isinstance(fs, pafs.FileSystem):
        return fs
    return pafs.PyFileSystem(pafs.FSSpecHandler(fs))


def arrow_filesystem(resource: ParquetDataResource) -> pafs.FileSystem | None:
    if uses_nonlocal_explicit_fs(resource):
        return _as_arrow_filesystem(resource.require_fs())
    return None


def normalized_load_path(resource: ParquetDataResource, path: str) -> str:
    if uses_nonlocal_explicit_fs(resource):
        return path
    return secure_local_path(resource, path)


def normalized_arrow_load_path(resource: ParquetDataResource, path: str) -> str:
    if uses_nonlocal_explicit_fs(resource):
        return strip_protocol(path)
    return secure_local_path(resource, path)


def strip_protocol(path: str) -> str:
    return path.split("://", 1)[1] if "://" in path else path


def restore_protocol(path: str) -> str:
    return str(path)
