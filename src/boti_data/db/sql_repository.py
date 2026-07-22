"""
Lightweight single-row transactional access (get/insert/update/delete by
primary key), returning plain dicts instead of a DataFrame — plus
SqlUnitOfWork/AsyncSqlUnitOfWork for composing writes across several tables
into one atomic transaction.

boti_data's flagship API (DataGateway/DataHelper) is dataframe-shaped and
bulk-oriented, pulling in the dask/pandas/polars dependency chain even for a
consumer that only ever touches one row at a time (e.g. an OLTP request
handler at p95 <300ms). SqlRepository/AsyncSqlRepository sit directly on
SqlDatabaseResource/AsyncSqlDatabaseResource + SqlModelRegistry — both already
free of that dependency chain — so this module can be imported and used
without dask/pandas/polars installed at all.
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import Any

from boti.core.lifecycle import LifecycleCore
from boti.core.logger import Logger
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.db.sql_model_registry import get_global_registry
from boti_data.db.sql_resource import AsyncSqlDatabaseResource, SqlDatabaseResource

__all__ = ["AsyncSqlRepository", "AsyncSqlUnitOfWork", "SqlRepository", "SqlUnitOfWork"]


def _row_to_dict(instance: Any) -> dict[str, Any]:
    mapper = sa_inspect(instance).mapper
    return {attr.key: getattr(instance, attr.key) for attr in mapper.column_attrs}


def _require_table_name(table_name: str | None) -> str:
    if table_name is None:
        raise ValueError("table_name is required.")
    return table_name


def _require_exactly_one_source(config: SqlDatabaseConfig | None, session: Any | None) -> None:
    if (config is None) == (session is None):
        raise ValueError("Provide exactly one of 'config' or 'session'.")


class SqlRepository(LifecycleCore):
    """Single-row get/insert/update/delete against one table, by primary key.

    A thin repository layer over SqlDatabaseResource + SqlModelRegistry for
    consumers that want cheap transactional CRUD without pulling in
    DataGateway/DataHelper's dataframe-oriented machinery. Rows are plain
    ``dict[str, Any]``, never a DataFrame.

    Exactly one of ``config``/``session`` must be given:

    - ``config``: the repository owns a private ``SqlDatabaseResource`` and a
      fresh, ``query_only``-guarded session per call, committing after every
      write. ``query_only`` defaults to ``True`` here and always wins over
      whatever ``config.query_only`` says — insert/update/delete raise via the
      same ``ReadOnlySession`` guard SqlDatabaseResource uses until a caller
      passes ``query_only=False`` explicitly. This is deliberate: a ``config``
      object may be shared with or reused from a context where writes are
      already enabled, and a repository shouldn't silently inherit write
      access nobody asked it for.
    - ``session``: an already-open ``Session`` supplied by the caller (e.g.
      from ``SqlUnitOfWork``, or hand-rolled session code composing several
      repositories into one transaction). Writes ``flush()`` instead of
      ``commit()`` — the caller controls the transaction boundary — and
      closing this repository never closes the shared session. ``query_only``
      is accepted for signature uniformity but is inert here: the session's
      own class (``ReadOnlySession`` or not) already determines writability,
      since this path never constructs a new guarded session.

    ``pk`` accepts a scalar or a tuple for composite keys, mirroring
    ``Session.get()``'s own signature.
    """

    def __init__(
        self,
        config: SqlDatabaseConfig | None = None,
        table_name: str | None = None,
        *,
        schema: str | None = None,
        query_only: bool = True,
        session: Session | None = None,
    ) -> None:
        table_name = _require_table_name(table_name)
        _require_exactly_one_source(config, session)
        self.logger = Logger.default_logger(logger_name=self.__class__.__name__)
        super().__init__()
        self._owns_resource = session is None
        self._session_override = session
        if session is not None:
            self._resource = None
            engine = session.bind
        else:
            effective_config = config.model_copy(update={"query_only": query_only})
            self._resource = SqlDatabaseResource(effective_config)
            engine = self._resource.engine
        self._model = get_global_registry().get_model(engine, table_name, schema=schema)

    def _cleanup(self) -> None:
        if self._resource is not None:
            self._resource.close()

    @property
    def engine(self) -> Any:
        return self._resource.engine if self._resource is not None else self._session_override.bind

    @contextmanager
    def _session_scope(self):
        if self._session_override is not None:
            yield self._session_override
        else:
            with self._resource.session() as session:
                yield session

    def _finalize_write(self, session: Session) -> None:
        if self._owns_resource:
            session.commit()
        else:
            session.flush()

    def get(self, pk: Any) -> dict[str, Any] | None:
        with self._session_scope() as session:
            instance = session.get(self._model, pk)
            return _row_to_dict(instance) if instance is not None else None

    def insert(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._session_scope() as session:
            instance = self._model(**values)
            session.add(instance)
            self._finalize_write(session)
            session.refresh(instance)
            return _row_to_dict(instance)

    def update(self, pk: Any, values: dict[str, Any]) -> dict[str, Any] | None:
        with self._session_scope() as session:
            instance = session.get(self._model, pk)
            if instance is None:
                return None
            for key, value in values.items():
                setattr(instance, key, value)
            self._finalize_write(session)
            session.refresh(instance)
            return _row_to_dict(instance)

    def delete(self, pk: Any) -> bool:
        with self._session_scope() as session:
            instance = session.get(self._model, pk)
            if instance is None:
                return False
            session.delete(instance)
            self._finalize_write(session)
            return True


class AsyncSqlRepository(LifecycleCore):
    """Async counterpart to :class:`SqlRepository` — see its docstring,
    including the ``config``/``session`` and ``query_only`` semantics.

    Async-only by design, mirroring ``AsyncSqlDatabaseResource``: real
    resource acquisition (the engine, the reflected model) happens in
    ``__aenter__``, not ``__init__``, since resolving the engine requires
    awaiting ``EngineRegistry.get_or_create_async()`` (``config`` path) or
    ``SqlModelRegistry.get_model_async()`` (``session`` path either way).
    """

    def __init__(
        self,
        config: SqlDatabaseConfig | None = None,
        table_name: str | None = None,
        *,
        schema: str | None = None,
        query_only: bool = True,
        session: AsyncSession | None = None,
    ) -> None:
        table_name = _require_table_name(table_name)
        _require_exactly_one_source(config, session)
        self.logger = Logger.default_logger(logger_name=self.__class__.__name__)
        self._config = (
            config.model_copy(update={"query_only": query_only}) if config is not None else None
        )
        self._table_name = table_name
        self._schema = schema
        self._session_override = session
        self._owns_resource = session is None
        self._resource: AsyncSqlDatabaseResource | None = None
        self._model: type[Any] | None = None
        super().__init__()

    async def __aenter__(self) -> AsyncSqlRepository:
        await super().__aenter__()
        if self._session_override is not None:
            engine = self._session_override.bind
        else:
            self._resource = await AsyncSqlDatabaseResource(self._config).__aenter__()
            engine = self._resource.engine
        self._model = await get_global_registry().get_model_async(
            engine, self._table_name, schema=self._schema
        )
        return self

    async def _acleanup(self) -> None:
        if self._resource is not None:
            await self._resource.aclose()

    @property
    def engine(self) -> Any:
        if self._resource is not None:
            return self._resource.engine
        if self._session_override is not None:
            return self._session_override.bind
        raise RuntimeError("AsyncSqlRepository is not bound — use 'async with'.")

    @asynccontextmanager
    async def _session_scope(self):
        if self._session_override is not None:
            yield self._session_override
        else:
            async with self._resource.session() as session:
                yield session

    async def _finalize_write(self, session: AsyncSession) -> None:
        if self._owns_resource:
            await session.commit()
        else:
            await session.flush()

    async def get(self, pk: Any) -> dict[str, Any] | None:
        async with self._session_scope() as session:
            instance = await session.get(self._model, pk)
            return _row_to_dict(instance) if instance is not None else None

    async def insert(self, values: dict[str, Any]) -> dict[str, Any]:
        async with self._session_scope() as session:
            instance = self._model(**values)
            session.add(instance)
            await self._finalize_write(session)
            await session.refresh(instance)
            return _row_to_dict(instance)

    async def update(self, pk: Any, values: dict[str, Any]) -> dict[str, Any] | None:
        async with self._session_scope() as session:
            instance = await session.get(self._model, pk)
            if instance is None:
                return None
            for key, value in values.items():
                setattr(instance, key, value)
            await self._finalize_write(session)
            await session.refresh(instance)
            return _row_to_dict(instance)

    async def delete(self, pk: Any) -> bool:
        async with self._session_scope() as session:
            instance = await session.get(self._model, pk)
            if instance is None:
                return False
            await session.delete(instance)
            await self._finalize_write(session)
            return True


class SqlUnitOfWork:
    """Shares one transaction across several ``SqlRepository`` instances,
    potentially against different tables, so their writes commit or roll back
    together.

    Not a ``LifecycleCore`` subclass: ``LifecycleCore.__exit__`` is marked
    ``@final`` specifically so cleanup-suppression logic can't be overridden,
    but that also means there's no hook that receives exception info to
    decide "commit on success, roll back on exception" — exactly what a unit
    of work needs. A small hand-rolled context manager is simpler than
    fighting that sealed contract.

    ``query_only`` defaults to ``True``, consistent with ``SqlRepository``'s
    own default — pass ``query_only=False`` explicitly to actually write.
    """

    def __init__(self, config: SqlDatabaseConfig, *, query_only: bool = True) -> None:
        effective_config = config.model_copy(update={"query_only": query_only})
        self._resource = SqlDatabaseResource(effective_config)
        self._session: Session = self._resource.session()
        self._repositories: list[SqlRepository] = []

    def repository(self, table_name: str, *, schema: str | None = None) -> SqlRepository:
        repo = SqlRepository(session=self._session, table_name=table_name, schema=schema)
        self._repositories.append(repo)
        return repo

    def __enter__(self) -> SqlUnitOfWork:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is None:
                self._session.commit()
            else:
                self._session.rollback()
        finally:
            # Each handed-out repository has its own GC leak-warning
            # finalizer (see boti.core.lifecycle) even though its own
            # _cleanup() is a no-op in session-injected mode — close them
            # here so that finalizer doesn't fire spuriously.
            for repo in self._repositories:
                repo.close()
            self._session.close()
            self._resource.close()


class AsyncSqlUnitOfWork:
    """Async counterpart to :class:`SqlUnitOfWork` — see its docstring.

    ``repository()`` is async here (unlike the sync version) because binding
    an ``AsyncSqlRepository`` to a shared session requires awaiting
    ``SqlModelRegistry.get_model_async()`` — the returned repository is
    already fully set up, so callers don't need to ``async with`` it too.
    """

    def __init__(self, config: SqlDatabaseConfig, *, query_only: bool = True) -> None:
        self._config = config.model_copy(update={"query_only": query_only})
        self._resource: AsyncSqlDatabaseResource | None = None
        self._session: AsyncSession | None = None
        self._repositories: list[AsyncSqlRepository] = []

    async def repository(self, table_name: str, *, schema: str | None = None) -> AsyncSqlRepository:
        repo = AsyncSqlRepository(session=self._session, table_name=table_name, schema=schema)
        await repo.__aenter__()
        self._repositories.append(repo)
        return repo

    async def __aenter__(self) -> AsyncSqlUnitOfWork:
        self._resource = await AsyncSqlDatabaseResource(self._config).__aenter__()
        self._session = self._resource.session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is None:
                await self._session.commit()
            else:
                await self._session.rollback()
        finally:
            # See SqlUnitOfWork.__exit__'s comment: each handed-out repository
            # has its own GC leak-warning finalizer even though its _acleanup()
            # is a no-op in session-injected mode.
            for repo in self._repositories:
                await repo.aclose()
            await self._session.close()
            await self._resource.aclose()
