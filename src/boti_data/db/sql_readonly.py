from __future__ import annotations

from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session


class ReadOnlySession(Session):
    """Session that blocks ORM mutations; raw SQL protection must come from the database."""

    def add(self, instance: object, _warn: bool = True) -> None:  # type: ignore[override]
        raise SQLAlchemyError("SqlDatabaseResource sessions are read-only.")

    def add_all(self, instances: Any) -> None:  # type: ignore[override]
        raise SQLAlchemyError("SqlDatabaseResource sessions are read-only.")

    def delete(self, instance: object) -> None:  # type: ignore[override]
        raise SQLAlchemyError("SqlDatabaseResource sessions are read-only.")

    def merge(self, instance: object, *, load: bool = True, options: Any = None) -> Any:  # type: ignore[override]
        raise SQLAlchemyError("SqlDatabaseResource sessions are read-only.")

    def flush(self, objects: Any = None) -> None:  # type: ignore[override]
        if objects is not None or self.new or self.dirty or self.deleted:
            raise SQLAlchemyError("SqlDatabaseResource sessions are read-only.")
        return super().flush(objects)

    def commit(self) -> None:
        raise SQLAlchemyError("SqlDatabaseResource sessions are read-only.")

    def bulk_insert_mappings(
        self,
        mapper: Any,
        mappings: Any,
        return_defaults: bool = False,
        render_nulls: bool = False,
    ) -> None:
        raise SQLAlchemyError("SqlDatabaseResource sessions are read-only.")

    def bulk_update_mappings(self, mapper: Any, mappings: Any) -> None:
        raise SQLAlchemyError("SqlDatabaseResource sessions are read-only.")

    def bulk_save_objects(
        self,
        objects: Any,
        return_defaults: bool = False,
        update_changed_only: bool = True,
        preserve_order: bool = True,
    ) -> None:
        raise SQLAlchemyError("SqlDatabaseResource sessions are read-only.")


class ReadOnlyAsyncSession(AsyncSession):
    """Async session that blocks ORM mutations; raw SQL protection must come from the database."""

    def add(self, instance: object, _warn: bool = True) -> None:  # type: ignore[override]
        raise SQLAlchemyError("AsyncSqlDatabaseResource sessions are read-only.")

    def add_all(self, instances: Any) -> None:  # type: ignore[override]
        raise SQLAlchemyError("AsyncSqlDatabaseResource sessions are read-only.")

    async def delete(self, instance: object) -> None:  # type: ignore[override]
        raise SQLAlchemyError("AsyncSqlDatabaseResource sessions are read-only.")

    async def merge(self, instance: object, *, load: bool = True, options: Any = None) -> Any:  # type: ignore[override]
        raise SQLAlchemyError("AsyncSqlDatabaseResource sessions are read-only.")

    async def flush(self, objects: Any = None) -> None:  # type: ignore[override]
        if objects is not None or self.new or self.dirty or self.deleted:
            raise SQLAlchemyError("AsyncSqlDatabaseResource sessions are read-only.")
        return await super().flush(objects)

    async def commit(self) -> None:
        raise SQLAlchemyError("AsyncSqlDatabaseResource sessions are read-only.")
