from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Protocol

import fsspec

_log = logging.getLogger(__name__)


class WatermarkStore(Protocol):
    """Protocol for persisting watermark values across incremental load runs."""

    def read(self, *, source: str) -> Any | None:
        """Return the persisted watermark for *source*, or ``None``."""
        ...

    def write(self, *, source: str, value: Any) -> None:
        """Persist *value* as the current watermark for *source*."""
        ...

    def clear(self, *, source: str) -> None:
        """Remove the persisted watermark for *source*."""
        ...


class FsspecWatermarkStore:
    """Thread-safe watermark store backed by a JSON file on any fsspec filesystem.

    Avoids importing ``fsspec`` at module level — the user passes an already-created
    filesystem instance.  This works with **any** fsspec implementation (local, S3,
    GCS, Azure, memory).

    Values are serialised as JSON so they must be JSON-serialisable (``str``,
    ``int``, ``float``, ``bool``, ``None``, or ISO-8601 strings for datetimes).

    Args:
        fs: An open fsspec filesystem instance.
        path: Path to the watermark JSON file on that filesystem.
    """

    def __init__(self, fs: fsspec.AbstractFileSystem, path: str) -> None:
        self._fs = fs
        self._path = path
        self._lock = threading.Lock()

    def read(self, *, source: str) -> Any | None:
        with self._lock:
            try:
                raw = self._fs.read_text(self._path)
            except FileNotFoundError:
                return None
            except Exception:
                _log.debug("Failed to read watermark file %s", self._path, exc_info=True)
                return None
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, OSError):
                return None
            return data.get(source)

    def _ensure_parent(self) -> None:
        self._fs.makedirs(os.path.dirname(self._path), exist_ok=True)

    def write(self, *, source: str, value: Any) -> None:
        with self._lock:
            try:
                raw = self._fs.read_text(self._path)
                data = json.loads(raw)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                data = {}
            data[source] = value
            self._ensure_parent()
            self._fs.write_text(self._path, json.dumps(data, default=str))

    def clear(self, *, source: str) -> None:
        with self._lock:
            try:
                raw = self._fs.read_text(self._path)
                data = json.loads(raw)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                return
            data.pop(source, None)
            self._ensure_parent()
            self._fs.write_text(self._path, json.dumps(data, default=str))


class FileWatermarkStore:
    """A watermark store backed by a local JSON file.

    Thread-safe via a per-instance lock.  Values are serialised as JSON
    so they must be JSON-serialisable (``str``, ``int``, ``float``,
    ``bool``, ``None``, or ISO-8601 strings for datetimes).

    Args:
        path: Filesystem path for the watermark JSON file.
    """

    def __init__(self, path: str = ".watermarks.json") -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def read(self, *, source: str) -> Any | None:
        with self._lock:
            if not self._path.exists():
                return None
            try:
                data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                return None
            return data.get(source)

    def write(self, *, source: str, value: Any) -> None:
        with self._lock:
            if self._path.exists():
                try:
                    data = json.loads(self._path.read_text())
                except (json.JSONDecodeError, OSError):
                    data = {}
            else:
                data = {}
            data[source] = value
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(data, default=str))

    def clear(self, *, source: str) -> None:
        with self._lock:
            if not self._path.exists():
                return
            try:
                data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                return
            data.pop(source, None)
            self._path.write_text(json.dumps(data, default=str))
