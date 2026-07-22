"""
Tests for pgvector Vector column reflection support.

boti_data.db.sql_model_registry optionally registers pgvector's 'vector' type
name into SQLAlchemy's Postgres dialect (if the pgvector package is installed),
so reflected Vector columns map to a usable type instead of NullType. There's
no portable, hermetic way to exercise the full reflect-a-real-table path here
(it needs a live Postgres with the vector extension installed), so this only
verifies the registration side effect itself — confirmed against a real
Postgres+pgvector instance during development.
"""

from sqlalchemy.dialects.postgresql.base import ischema_names


def test_pgvector_type_is_registered_for_reflection() -> None:
    # Importing boti_data.db.sql_model_registry (already imported by the time
    # this test runs, via boti_data.db) must have registered pgvector's Vector
    # type, given pgvector is installed in the dev/test environment.
    import pgvector.sqlalchemy

    assert ischema_names.get("vector") is pgvector.sqlalchemy.Vector
