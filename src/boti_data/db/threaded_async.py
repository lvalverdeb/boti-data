"""Executor-backed asyncio facade over a synchronous SQLAlchemy engine.

DuckDB (``duckdb-engine``) and ClickHouse (``clickhouse-connect``) ship
synchronous SQLAlchemy dialects only, so SQLAlchemy's own asyncio extension
cannot wrap them: ``create_async_engine`` and ``AsyncEngine(sync_engine)`` both
refuse a dialect whose ``is_async`` is false. This module offers the same
``engine``/``session`` shape ``AsyncSqlDatabaseResource`` exposes for native
async drivers, with every blocking call run in a worker thread through
``asyncio.to_thread``.

Surface offered: ``engine.connect()``, ``engine.begin()``, ``engine.dispose()``,
``engine.url``/``dialect``/``sync_engine``; on a connection ``execute``,
``exec_driver_sql``, ``scalar``, ``run_sync``, ``commit``, ``rollback``,
``close``; on a session ``execute``, ``scalar``, ``scalars``, ``get``, ``add``,
``add_all``, ``delete``, ``merge``, ``flush``, ``refresh``, ``commit``,
``rollback``, ``close``, ``run_sync``. Not offered: ``stream()`` (server-side
async cursors need a native async driver) and ``AsyncEngine``-only internals
such as ``engine.raw_connection()``; ``isinstance(x, AsyncSession)`` is false
for these sessions.

Writer serialisation: pass ``write_lock`` and every statement on every
connection and session of that engine runs under it, one at a time. DuckDB
needs this because a database file admits one writing process and concurrent
writers in one process surface as transaction conflicts; with
``query_only=True`` no lock is passed and reads run concurrently.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

__all__ = [
    "ThreadedAsyncConnection",
    "ThreadedAsyncEngine",
    "ThreadedAsyncSession",
    "threaded_async_sessionmaker",
]


class _Threaded:
    """Run a sync callable in a worker thread, under the writer lock when set."""

    __slots__ = ("_write_lock",)

    def __init__(self, write_lock: asyncio.Lock | None) -> None:
        self._write_lock = write_lock

    async def _run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._write_lock is None:
            return await asyncio.to_thread(fn, *args, **kwargs)
        async with self._write_lock:
            return await asyncio.to_thread(fn, *args, **kwargs)


class ThreadedAsyncConnection(_Threaded):
    """``AsyncConnection``-shaped facade over a sync ``Connection``."""

    __slots__ = ("sync_connection",)

    def __init__(self, sync_connection: Any, write_lock: asyncio.Lock | None) -> None:
        super().__init__(write_lock)
        self.sync_connection = sync_connection

    async def execute(self, statement: Any, parameters: Any = None, **kwargs: Any) -> Any:
        return await self._run(self.sync_connection.execute, statement, parameters, **kwargs)

    async def exec_driver_sql(self, statement: str, parameters: Any = None, **kwargs: Any) -> Any:
        return await self._run(
            self.sync_connection.exec_driver_sql, statement, parameters, **kwargs
        )

    async def scalar(self, statement: Any, parameters: Any = None, **kwargs: Any) -> Any:
        return await self._run(self.sync_connection.scalar, statement, parameters, **kwargs)

    async def run_sync(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return await self._run(fn, self.sync_connection, *args, **kwargs)

    async def commit(self) -> None:
        await self._run(self.sync_connection.commit)

    async def rollback(self) -> None:
        await self._run(self.sync_connection.rollback)

    async def close(self) -> None:
        await self._run(self.sync_connection.close)

    @property
    def closed(self) -> bool:
        return bool(self.sync_connection.closed)


class ThreadedAsyncEngine:
    """``AsyncEngine``-shaped facade over a sync ``Engine``; see the module docstring."""

    def __init__(self, sync_engine: Engine, *, write_lock: asyncio.Lock | None = None) -> None:
        self.sync_engine = sync_engine
        self._write_lock = write_lock

    @property
    def url(self) -> Any:
        return self.sync_engine.url

    @property
    def dialect(self) -> Any:
        return self.sync_engine.dialect

    @property
    def write_lock(self) -> asyncio.Lock | None:
        return self._write_lock

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[ThreadedAsyncConnection]:
        connection = ThreadedAsyncConnection(
            await asyncio.to_thread(self.sync_engine.connect), self._write_lock
        )
        try:
            yield connection
        finally:
            await connection.close()

    @contextlib.asynccontextmanager
    async def begin(self) -> AsyncIterator[ThreadedAsyncConnection]:
        """Open a connection in a transaction, committed on clean exit, rolled back on error."""
        async with self.connect() as connection:
            try:
                yield connection
            except BaseException:
                await connection.rollback()
                raise
            await connection.commit()

    async def dispose(self) -> None:
        await asyncio.to_thread(self.sync_engine.dispose)


class ThreadedAsyncSession(_Threaded):
    """``AsyncSession``-shaped facade over a sync ORM ``Session``.

    ``add``/``add_all`` are synchronous, as on ``AsyncSession``. Whether the
    wrapped session is read-only (``ReadOnlySession``) is visible through
    :attr:`sync_session`, which :func:`boti_data.db.sql_readonly.is_read_only_session`
    inspects.
    """

    __slots__ = ("_bind", "sync_session")

    def __init__(self, sync_session: Session, bind: ThreadedAsyncEngine) -> None:
        super().__init__(bind.write_lock)
        self.sync_session = sync_session
        self._bind = bind

    @property
    def bind(self) -> ThreadedAsyncEngine:
        """The async facade, so code given this session reflects through it."""
        return self._bind

    async def __aenter__(self) -> ThreadedAsyncSession:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    # spaghetti-ignore[pass-through-method]: AsyncSession's add() is sync by contract; the facade mirrors it
    def add(self, instance: object, _warn: bool = True) -> None:
        self.sync_session.add(instance, _warn=_warn)

    # spaghetti-ignore[pass-through-method]: as add()
    def add_all(self, instances: Any) -> None:
        self.sync_session.add_all(instances)

    async def execute(self, statement: Any, params: Any = None, **kwargs: Any) -> Any:
        return await self._run(self.sync_session.execute, statement, params, **kwargs)

    async def scalar(self, statement: Any, params: Any = None, **kwargs: Any) -> Any:
        return await self._run(self.sync_session.scalar, statement, params, **kwargs)

    async def scalars(self, statement: Any, params: Any = None, **kwargs: Any) -> Any:
        return await self._run(self.sync_session.scalars, statement, params, **kwargs)

    async def get(self, entity: Any, ident: Any, **kwargs: Any) -> Any:
        return await self._run(self.sync_session.get, entity, ident, **kwargs)

    async def delete(self, instance: object) -> None:
        await self._run(self.sync_session.delete, instance)

    async def merge(self, instance: Any, **kwargs: Any) -> Any:
        return await self._run(self.sync_session.merge, instance, **kwargs)

    async def flush(self, objects: Any = None) -> None:
        await self._run(self.sync_session.flush, objects)

    async def refresh(self, instance: object, **kwargs: Any) -> None:
        await self._run(self.sync_session.refresh, instance, **kwargs)

    async def commit(self) -> None:
        await self._run(self.sync_session.commit)

    async def rollback(self) -> None:
        await self._run(self.sync_session.rollback)

    async def close(self) -> None:
        await self._run(self.sync_session.close)

    async def run_sync(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return await self._run(fn, self.sync_session, *args, **kwargs)


class threaded_async_sessionmaker:  # noqa: N801 - mirrors sqlalchemy's async_sessionmaker
    """Callable factory producing :class:`ThreadedAsyncSession`, like ``async_sessionmaker``."""

    def __init__(
        self,
        bind: ThreadedAsyncEngine,
        *,
        class_: type[Session] = Session,
        **session_kwargs: Any,
    ) -> None:
        self._factory = sessionmaker(bind=bind.sync_engine, class_=class_, **session_kwargs)
        self._bind = bind

    def __call__(self, **kwargs: Any) -> ThreadedAsyncSession:
        return ThreadedAsyncSession(self._factory(**kwargs), self._bind)
