"""
pgvector Vector column reflection + nearest-neighbour search example.

Needs a real Postgres instance with the pgvector extension available (SQLite
can't do this) and boti-data[pgvector] + boti-data[postgres] installed.
Skips gracefully — prints a message and exits 0 — if either the optional
dependencies or a reachable pgvector-enabled Postgres aren't available, so
this doesn't fail examples/smoke_all_examples.py on a machine without both.

Point BOTI_EXAMPLE_PGVECTOR_DSN at your own instance if the default
(postgresql+psycopg://postgres:postgres@localhost:5432/postgres) doesn't
match your setup.
"""

from __future__ import annotations

import os

from sqlalchemy import Column, Integer, MetaData, Table, text
from sqlalchemy.exc import SQLAlchemyError

DSN = os.environ.get(
    "BOTI_EXAMPLE_PGVECTOR_DSN",
    "postgresql+psycopg://postgres:postgres@localhost:5432/postgres",
)


def _run_demo(config) -> None:
    import pgvector.sqlalchemy

    from boti_data.db import SqlAlchemyModelBuilder, SqlDatabaseResource, nearest_neighbors

    with SqlDatabaseResource(config) as db:
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        metadata = MetaData()
        Table(
            "boti_pgvector_example",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("embedding", pgvector.sqlalchemy.Vector(3)),
        )
        with db.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            metadata.drop_all(conn, checkfirst=True)
            metadata.create_all(conn)

        try:
            model = SqlAlchemyModelBuilder(db.engine, "boti_pgvector_example").build_model()
            print(f"reflected_type={model.__table__.columns['embedding'].type!r}")

            with db.session() as session:
                session.add_all(
                    [
                        model(id=1, embedding=[1.0, 0.0, 0.0]),
                        model(id=2, embedding=[0.9, 0.1, 0.0]),
                        model(id=3, embedding=[0.0, 1.0, 0.0]),
                    ]
                )
                session.commit()

                query_vector = [1.0, 0.0, 0.0]
                stmt = nearest_neighbors(model, model.embedding, query_vector, k=2, metric="cosine")
                nearest = session.execute(stmt).scalars().all()
                print(f"nearest_ids={[row.id for row in nearest]}")
        finally:
            with db.engine.begin() as conn:
                metadata.drop_all(conn, checkfirst=True)


def main() -> None:
    try:
        import pgvector.sqlalchemy  # noqa: F401  registers Vector for reflection
    except ImportError:
        print("skipped: pgvector is not installed (pip install 'boti-data[pgvector]')")
        return

    from boti_data.db import SqlDatabaseConfig

    config = SqlDatabaseConfig(connection_url=DSN, query_only=False)

    try:
        _run_demo(config)
    except (SQLAlchemyError, ImportError, OSError) as exc:
        print(f"skipped: pgvector-enabled Postgres not reachable at {DSN!r} ({exc})")


if __name__ == "__main__":
    main()
