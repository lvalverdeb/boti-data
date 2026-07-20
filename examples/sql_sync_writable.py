"""
Synchronous SQL example with query_only explicitly disabled.
"""

from __future__ import annotations

from sqlalchemy import text

from boti_data.db import SqlDatabaseConfig, SqlDatabaseResource


def main() -> None:
    config = SqlDatabaseConfig(
        connection_url="sqlite:///:memory:",
        poolclass="sqlalchemy.pool.StaticPool",
        query_only=False,
    )

    with SqlDatabaseResource(config) as first, SqlDatabaseResource(config) as second:
        print(f"shared_engine={first.engine is second.engine}")

        with first.engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE writable_users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.exec_driver_sql("INSERT INTO writable_users (id, name) VALUES (1, 'Alice')")

        with second.session() as session:
            names = (
                session.execute(text("SELECT name FROM writable_users ORDER BY id")).scalars().all()
            )

        print(f"rows={names}")


if __name__ == "__main__":
    main()
