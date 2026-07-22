"""
Lightweight single-row transactional access (get/insert/update/delete by
primary key), returning plain dicts instead of a DataFrame.

boti_data's flagship API (DataGateway/DataHelper) is dataframe-shaped and
bulk-oriented, pulling in the dask/pandas/polars dependency chain even for a
consumer that only ever touches one row at a time (e.g. an OLTP request
handler at p95 <300ms). SqlRepository/AsyncSqlRepository sit directly on
SqlDatabaseResource/AsyncSqlDatabaseResource + SqlModelRegistry — both already
free of that dependency chain — so this module can be imported and used
without dask/pandas/polars installed at all.
"""

from __future__ import annotations

from typing import Any

from boti.core.lifecycle import LifecycleCore
from boti.core.logger import Logger
from sqlalchemy import inspect as sa_inspect

from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.db.sql_model_registry import get_global_registry
from boti_data.db.sql_resource import AsyncSqlDatabaseResource, SqlDatabaseResource

__all__ = ["AsyncSqlRepository", "SqlRepository"]


def _row_to_dict(instance: Any) -> dict[str, Any]:
    mapper = sa_inspect(instance).mapper
    return {attr.key: getattr(instance, attr.key) for attr in mapper.column_attrs}


class SqlRepository(LifecycleCore):
    """Single-row get/insert/update/delete against one table, by primary key.

    A thin repository layer over SqlDatabaseResource + SqlModelRegistry for
    consumers that want cheap transactional CRUD without pulling in
    DataGateway/DataHelper's dataframe-oriented machinery. Rows are plain
    ``dict[str, Any]``, never a DataFrame.

    ``query_only`` defaults to ``True`` here and always wins over whatever
    ``config.query_only`` says — insert/update/delete raise via the same
    ``ReadOnlySession`` guard SqlDatabaseResource uses until a caller passes
    ``query_only=False`` explicitly. This is deliberate: a ``config`` object
    may be shared with or reused from a context where writes are already
    enabled, and a repository shouldn't silently inherit write access nobody
    asked it for.

    ``pk`` accepts a scalar or a tuple for composite keys, mirroring
    ``Session.get()``'s own signature.
    """

    def __init__(
        self,
        config: SqlDatabaseConfig,
        table_name: str,
        *,
        schema: str | None = None,
        query_only: bool = True,
    ) -> None:
        self.logger = Logger.default_logger(logger_name=self.__class__.__name__)
        super().__init__()
        effective_config = config.model_copy(update={"query_only": query_only})
        self._resource = SqlDatabaseResource(effective_config)
        self._model = get_global_registry().get_model(
            self._resource.engine, table_name, schema=schema
        )

    def _cleanup(self) -> None:
        self._resource.close()

    @property
    def engine(self) -> Any:
        return self._resource.engine

    def get(self, pk: Any) -> dict[str, Any] | None:
        with self._resource.session() as session:
            instance = session.get(self._model, pk)
            return _row_to_dict(instance) if instance is not None else None

    def insert(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._resource.session() as session:
            instance = self._model(**values)
            session.add(instance)
            session.commit()
            session.refresh(instance)
            return _row_to_dict(instance)

    def update(self, pk: Any, values: dict[str, Any]) -> dict[str, Any] | None:
        with self._resource.session() as session:
            instance = session.get(self._model, pk)
            if instance is None:
                return None
            for key, value in values.items():
                setattr(instance, key, value)
            session.commit()
            session.refresh(instance)
            return _row_to_dict(instance)

    def delete(self, pk: Any) -> bool:
        with self._resource.session() as session:
            instance = session.get(self._model, pk)
            if instance is None:
                return False
            session.delete(instance)
            session.commit()
            return True


class AsyncSqlRepository(LifecycleCore):
    """Async counterpart to :class:`SqlRepository` — see its docstring,
    including the ``query_only`` default/override behaviour.

    Async-only by design, mirroring ``AsyncSqlDatabaseResource``: real
    resource acquisition (the engine, the reflected model) happens in
    ``__aenter__``, not ``__init__``, since resolving the engine requires
    awaiting ``EngineRegistry.get_or_create_async()``.
    """

    def __init__(
        self,
        config: SqlDatabaseConfig,
        table_name: str,
        *,
        schema: str | None = None,
        query_only: bool = True,
    ) -> None:
        self.logger = Logger.default_logger(logger_name=self.__class__.__name__)
        self._config = config.model_copy(update={"query_only": query_only})
        self._table_name = table_name
        self._schema = schema
        self._resource: AsyncSqlDatabaseResource | None = None
        self._model: type[Any] | None = None
        super().__init__()

    async def __aenter__(self) -> AsyncSqlRepository:
        await super().__aenter__()
        self._resource = await AsyncSqlDatabaseResource(self._config).__aenter__()
        self._model = await get_global_registry().get_model_async(
            self._resource.engine, self._table_name, schema=self._schema
        )
        return self

    async def _acleanup(self) -> None:
        if self._resource is not None:
            await self._resource.aclose()

    @property
    def engine(self) -> Any:
        if self._resource is None:
            raise RuntimeError("AsyncSqlRepository is not bound — use 'async with'.")
        return self._resource.engine

    async def get(self, pk: Any) -> dict[str, Any] | None:
        async with self._resource.session() as session:
            instance = await session.get(self._model, pk)
            return _row_to_dict(instance) if instance is not None else None

    async def insert(self, values: dict[str, Any]) -> dict[str, Any]:
        async with self._resource.session() as session:
            instance = self._model(**values)
            session.add(instance)
            await session.commit()
            await session.refresh(instance)
            return _row_to_dict(instance)

    async def update(self, pk: Any, values: dict[str, Any]) -> dict[str, Any] | None:
        async with self._resource.session() as session:
            instance = await session.get(self._model, pk)
            if instance is None:
                return None
            for key, value in values.items():
                setattr(instance, key, value)
            await session.commit()
            await session.refresh(instance)
            return _row_to_dict(instance)

    async def delete(self, pk: Any) -> bool:
        async with self._resource.session() as session:
            instance = await session.get(self._model, pk)
            if instance is None:
                return False
            await session.delete(instance)
            await session.commit()
            return True
