"""
Synchronous SQL example using the default query-only mode.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from boti_data.db import SqlDatabaseConfig, SqlDatabaseResource


def seed_database(db_path: Path) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE readonly_users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.exec_driver_sql("INSERT INTO readonly_users (name) VALUES ('Alice')")
    engine.dispose()


def main() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "readonly.db"
        seed_database(db_path)

        config = SqlDatabaseConfig(
            connection_url=f"sqlite:///file:{db_path}?mode=ro&uri=true",
            poolclass="sqlalchemy.pool.NullPool",
        )

        with SqlDatabaseResource(config) as db:
            with db.engine.connect() as conn:
                value = conn.execute(text("SELECT name FROM readonly_users")).scalar_one()
                print(f"selected={value}")

                try:
                    conn.exec_driver_sql("INSERT INTO readonly_users (name) VALUES ('Bob')")
                except OperationalError as exc:
                    print(f"write_blocked={type(exc).__name__}")

            with db.session() as session:
                try:
                    session.commit()
                except SQLAlchemyError as exc:
                    print(f"session_commit_blocked={exc}")


if __name__ == "__main__":
    main()
