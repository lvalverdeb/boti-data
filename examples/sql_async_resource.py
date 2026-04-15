"""
Async SQL example.

Set BOTI_EXAMPLE_ASYNC_DB_URL to a reachable async DSN before running, for example:
postgresql+asyncpg://user:pass@localhost/dbname
"""

from __future__ import annotations

import asyncio
import os

from sqlalchemy import text

from boti_data.db import AsyncSqlDatabaseResource, SqlDatabaseConfig, ensure_greenlet_available


async def main() -> None:
    dsn = os.environ.get("BOTI_EXAMPLE_ASYNC_DB_URL")
    if not dsn:
        print("Set BOTI_EXAMPLE_ASYNC_DB_URL to a reachable async database DSN.")
        return

    ensure_greenlet_available()
    config = SqlDatabaseConfig(connection_url=dsn)

    async with AsyncSqlDatabaseResource(config) as db:
        async with db.session() as session:
            result = await session.execute(text("SELECT 1"))
            print(f"selected={result.scalar_one()}")


if __name__ == "__main__":
    asyncio.run(main())
