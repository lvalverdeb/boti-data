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


class _JsonWatermarkStoreBase:
    """Shared read/write/clear orchestration for JSON-file-backed watermark stores.

    Subclasses provide storage-specific ``_read_payload``/``_write_payload`` hooks
    and must set ``self._lock`` in ``__init__``.
    """

    _lock: threading.Lock

    def _read_payload(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def _write_payload(self, data: dict[str, Any]) -> None:
        raise NotImplementedError

    def read(self, *, source: str) -> Any | None:
        with self._lock:
            data = self._read_payload()
            return None if data is None else data.get(source)

    def write(self, *, source: str, value: Any) -> None:
        with self._lock:
            data = self._read_payload() or {}
            data[source] = value
            self._write_payload(data)

    def clear(self, *, source: str) -> None:
        with self._lock:
            data = self._read_payload()
            if data is None:
                return
            data.pop(source, None)
            self._write_payload(data)


class FsspecWatermarkStore(_JsonWatermarkStoreBase):
    """Thread-safe watermark store backed by a JSON file on any fsspec filesystem.

    Avoids importing ``fsspec`` at module level — the user passes an already-created
    filesystem instance.  This works with **any** fsspec implementation (local, S3,
    GCS, Azure, memory).

    Values are serialised as JSON so they must be JSON-serialisable (``str``,
    ``int``, ``float``, ``bool``, ``None``, or ISO-8601 strings for datetimes).

    Thread-safe within one process only.  Concurrent writers in *different*
    processes are last-writer-wins: coordinate externally (or use one store
    file per pipeline) if multiple processes update the same file.

    Args:
        fs: An open fsspec filesystem instance.
        path: Path to the watermark JSON file on that filesystem.
    """

    def __init__(self, fs: fsspec.AbstractFileSystem, path: str) -> None:
        self._fs = fs
        self._path = path
        self._lock = threading.Lock()

    def _ensure_parent(self) -> None:
        self._fs.makedirs(os.path.dirname(self._path), exist_ok=True)

    def _read_raw_text(self) -> str | None:
        try:
            return self._fs.read_text(self._path)
        except FileNotFoundError:
            return None
        except Exception:
            _log.debug("Failed to read watermark file %s", self._path, exc_info=True)
            return None

    def _read_payload(self) -> dict[str, Any] | None:
        raw = self._read_raw_text()
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, OSError):
            # A corrupt watermark file silently turns every incremental load
            # into a full reload — make sure operators can see why.
            _log.warning(
                "Watermark file %s is corrupt; treating all watermarks as missing.",
                self._path,
            )
            return None

    def _write_payload(self, data: dict[str, Any]) -> None:
        """Persist atomically where the filesystem allows it.

        Writes to a sibling temp file and renames over the target, so a crash
        mid-write cannot leave a truncated JSON file. Falls back to a direct
        write on filesystems where rename-over is unsupported.
        """
        payload = json.dumps(data, default=str)
        self._ensure_parent()
        tmp_path = f"{self._path}.tmp"
        try:
            self._fs.write_text(tmp_path, payload)
            self._fs.mv(tmp_path, self._path)
        except Exception:
            _log.debug(
                "Atomic rename unsupported for %s; falling back to direct write.",
                self._path,
                exc_info=True,
            )
            try:
                self._fs.rm_file(tmp_path)
            except Exception:
                _log.debug("Failed to remove temp watermark file %s", tmp_path, exc_info=True)
            self._fs.write_text(self._path, payload)


class FileWatermarkStore(_JsonWatermarkStoreBase):
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

    def _read_payload(self) -> dict[str, Any] | None:
        if not self._path.exists():
            return None
        try:
            return json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            _log.warning(
                "Watermark file %s is corrupt; treating all watermarks as missing.",
                self._path,
            )
            return None

    def _write_payload(self, data: dict[str, Any]) -> None:
        # Temp file + os.replace: atomic on POSIX, so a crash mid-write can
        # never leave a truncated JSON file behind.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, default=str))
        os.replace(tmp_path, self._path)


class SqlWatermarkStore:
    """A watermark store kept in the database next to the data.

    Uses one small table, ``boti_watermarks(source TEXT PRIMARY KEY, value TEXT,
    updated_at TEXT)``, created on first use. Values are JSON-encoded exactly as
    the file-backed stores encode them, so a watermark can move between stores.
    Every operation is one short transaction on the given engine.

    Args:
        engine: A SQLAlchemy engine (sync) with write access.
        table_name: Override for the bookkeeping table name.
    """

    def __init__(self, engine: Any, *, table_name: str = "boti_watermarks") -> None:
        import sqlalchemy as sa

        self._engine = engine
        self._table = sa.Table(
            table_name,
            sa.MetaData(),
            sa.Column("source", sa.String(255), primary_key=True),
            sa.Column("value", sa.Text, nullable=True),
            sa.Column("updated_at", sa.String(64), nullable=False),
        )
        self._lock = threading.Lock()
        self._created = False

    def _ensure_table(self) -> None:
        if not self._created:
            self._table.create(self._engine, checkfirst=True)
            self._created = True

    def read(self, *, source: str) -> Any | None:
        import sqlalchemy as sa

        with self._lock:
            self._ensure_table()
            with self._engine.connect() as conn:
                row = conn.execute(
                    sa.select(self._table.c.value).where(self._table.c.source == source)
                ).first()
        return None if row is None or row[0] is None else json.loads(row[0])

    def write(self, *, source: str, value: Any) -> None:
        import datetime as dt

        import sqlalchemy as sa

        payload = json.dumps(value, default=str)
        stamp = dt.datetime.now(dt.UTC).isoformat()
        with self._lock:
            self._ensure_table()
            with self._engine.begin() as conn:
                updated = conn.execute(
                    sa.update(self._table)
                    .where(self._table.c.source == source)
                    .values(value=payload, updated_at=stamp)
                ).rowcount
                if not updated:
                    conn.execute(
                        sa.insert(self._table).values(
                            source=source, value=payload, updated_at=stamp
                        )
                    )

    def clear(self, *, source: str) -> None:
        import sqlalchemy as sa

        with self._lock:
            self._ensure_table()
            with self._engine.begin() as conn:
                conn.execute(sa.delete(self._table).where(self._table.c.source == source))
