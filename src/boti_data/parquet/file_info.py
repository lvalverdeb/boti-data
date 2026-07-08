from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import pyarrow.fs as pafs

FileInfo = dict[str, Any]
MissingPathPredicate = Callable[[Exception], bool]
RuntimeErrorFactory = Callable[[str, str, Exception], RuntimeError]


class FileInfoProvider(Protocol):
    def info(self, path: str) -> FileInfo | None:
        raise NotImplementedError


class ArrowFileInfoProvider:
    def __init__(
        self,
        filesystem: pafs.FileSystem,
        *,
        is_missing_path_error: MissingPathPredicate,
        filesystem_runtime_error: RuntimeErrorFactory,
    ) -> None:
        self._filesystem = filesystem
        self._is_missing_path_error = is_missing_path_error
        self._filesystem_runtime_error = filesystem_runtime_error

    def info(self, path: str) -> FileInfo | None:
        try:
            file_info = self._filesystem.get_file_info(path)
        except Exception as exc:
            if self._is_missing_path_error(exc):
                return None
            raise self._filesystem_runtime_error("read parquet file metadata", path, exc) from exc

        if getattr(file_info, "type", None) == pafs.FileType.NotFound:
            return None
        return self._payload_from_arrow_info(file_info)

    @staticmethod
    def _payload_from_arrow_info(file_info: Any) -> FileInfo:
        payload: FileInfo = {}
        size = getattr(file_info, "size", None)
        if isinstance(size, int):
            payload["size"] = size
        modified = getattr(file_info, "mtime", None)
        if modified is not None:
            payload["mtime"] = modified
        return payload


class FsspecFileInfoProvider:
    def __init__(
        self,
        filesystem: Any,
        *,
        is_missing_path_error: MissingPathPredicate,
        filesystem_runtime_error: RuntimeErrorFactory,
    ) -> None:
        self._filesystem = filesystem
        self._is_missing_path_error = is_missing_path_error
        self._filesystem_runtime_error = filesystem_runtime_error

    def info(self, path: str) -> FileInfo | None:
        try:
            return self._filesystem.info(path)
        except Exception as exc:
            if self._is_missing_path_error(exc):
                return None
            return self._try_modified(path, exc)

    def _try_modified(self, path: str, original_exc: Exception) -> FileInfo | None:
        if not hasattr(self._filesystem, "modified"):
            raise self._filesystem_runtime_error(
                "read parquet file metadata",
                path,
                original_exc,
            ) from original_exc

        try:
            modified = self._filesystem.modified(path)
        except Exception as exc:
            if self._is_missing_path_error(exc):
                return None
            raise self._filesystem_runtime_error("read parquet file metadata", path, exc) from exc

        return {"mtime": modified} if modified is not None else None
