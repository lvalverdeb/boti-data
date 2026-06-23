from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Protocol


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
