"""Shared SQLAlchemy models and DataHelper/HybridDataset builders for the pipelines tests.

Split out of test_pipelines.py purely for god-module/long-file headroom.
Leading underscore so pytest does not collect this as a test module.
"""

from __future__ import annotations

import datetime as dt

from fsspec.spec import AbstractFileSystem
from sqlalchemy import Date, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataHelper, HybridDataset


class Base(DeclarativeBase):
    pass


class SourceEvent(Base):
    __tablename__ = "source_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(32))


class HistoricalEvent(Base):
    __tablename__ = "historical_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(32))


class LiveEvent(Base):
    __tablename__ = "live_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(32))


def _build_source_helper(tmp_path) -> DataHelper:
    db_path = tmp_path / "pipeline_source.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    SourceEvent(id=1, event_date=dt.date(2026, 4, 15), status="active"),
                    SourceEvent(id=2, event_date=dt.date(2026, 4, 16), status="inactive"),
                    SourceEvent(id=3, event_date=dt.date(2026, 4, 17), status="active"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    return DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="source_events",
    )


def _build_hybrid_dataset(tmp_path) -> HybridDataset:
    db_path = tmp_path / "pipeline_hybrid.db"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    HistoricalEvent(id=1, event_date=dt.date(2026, 4, 15), status="hist"),
                    HistoricalEvent(id=2, event_date=dt.date(2026, 4, 17), status="hist"),
                    LiveEvent(id=10, event_date=dt.date(2026, 4, 18), status="live"),
                    LiveEvent(id=11, event_date=dt.date(2026, 4, 20), status="live"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()

    historical_helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="historical_events",
    )
    live_helper = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="live_events",
    )
    return HybridDataset(
        historical_helper,
        live_helper,
        date_field="event_date",
        split_date="2026-04-18",
    )


class PrefixOnlyFakeFileSystem(AbstractFileSystem):
    """In-memory fsspec filesystem modelling a prefix-only S3-compatible store.

    Unlike a real directory, no object exists at a bare "directory" key --
    only leaf object keys are real. ``find(..., withdirs=True)`` on such a
    backend can surface a synthetic entry for the bare prefix itself even
    though nothing exists there, which is exactly the wishlist #1 MinIO
    reproduction: ``expand_path(recursive=True)`` returning a phantom
    directory "file" alongside the real objects.
    """

    protocol = "prefixonly"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.store: dict[str, bytes] = {}

    def _strip_protocol(self, path):
        return str(path).rstrip("/")

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        prefix = path + "/"
        names = sorted(key for key in self.store if key == path or key.startswith(prefix))
        if detail:
            return [{"name": name, "type": "file", "size": len(self.store[name])} for name in names]
        return names

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        path = self._strip_protocol(path)
        prefix = path + "/"
        keys = sorted(key for key in self.store if key == path or key.startswith(prefix))
        if withdirs and path not in keys:
            keys = [path, *keys]
        return keys

    def exists(self, path, **kwargs) -> bool:
        path = self._strip_protocol(path)
        prefix = path + "/"
        return path in self.store or any(key.startswith(prefix) for key in self.store)

    def isdir(self, path) -> bool:
        path = self._strip_protocol(path)
        return path not in self.store and self.exists(path)

    def cp_file(self, path1, path2, **kwargs) -> None:
        path1 = self._strip_protocol(path1)
        path2 = self._strip_protocol(path2)
        if path1 not in self.store:
            raise FileNotFoundError(path1)
        self.store[path2] = self.store[path1]

    def rm(self, path, recursive=False, **kwargs) -> None:
        path = self._strip_protocol(path)
        if recursive:
            prefix = path + "/"
            for key in [k for k in self.store if k == path or k.startswith(prefix)]:
                del self.store[key]
        else:
            self.store.pop(path, None)

    def makedirs(self, path, exist_ok=True) -> None:
        pass

    def mkdir(self, path, **kwargs) -> None:
        pass
