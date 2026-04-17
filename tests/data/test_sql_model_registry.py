"""
Tests for the dynamic SQLAlchemy model registry logic.
"""
from __future__ import annotations

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine
from sqlalchemy.exc import SQLAlchemyError

from boti_data.db.sql_model_registry import RegistryConfig, SqlModelRegistry


def test_registry_dynamic_model_generation():
    engine = create_engine("sqlite:///:memory:")

    # Create the db schema manually
    metadata = MetaData()
    test_table = Table(
        'test_dynamic_users', metadata,
        Column('id', Integer, primary_key=True),
        Column('name', String)
    )
    metadata.create_all(engine)

    registry = SqlModelRegistry()

    # Generate model via registry
    UserModel = registry.get_model(engine, "test_dynamic_users")

    assert UserModel.__name__ == "TestDynamicUsers"
    assert UserModel.__tablename__ == "test_dynamic_users"

    # Test identical model reference is returned from cache on exact subsequent call
    UserModel2 = registry.get_model(engine, "test_dynamic_users")
    assert UserModel is UserModel2


def test_registry_missing_pk_fallback():
    """Verify registry dynamically falls back to sequential columns when PK is missing."""
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()

    Table(
        'missing_pk_table', metadata,
        Column('pseudo_id', Integer),
        Column('data', String)
    )
    metadata.create_all(engine)

    messages: list[str] = []

    class StubLogger:
        def warning(self, message: str) -> None:
            messages.append(message)

    registry = SqlModelRegistry(logger=StubLogger())  # type: ignore[arg-type]
    Model = registry.get_model(engine, "missing_pk_table")

    # Should resolve by injecting primary column
    assert any("Missing native Primary Key" in message for message in messages)
    assert Model.__mapper_args__["primary_key"][0].name == "pseudo_id"


@pytest.mark.asyncio
async def test_async_registry_requires_greenlet_before_engine_use(monkeypatch):
    """Verify async registry surfaces a clear dependency error."""
    monkeypatch.setattr("boti_data.db.sql_model_registry.ensure_greenlet_available", lambda: (_ for _ in ()).throw(SQLAlchemyError("missing greenlet")))

    registry = SqlModelRegistry()

    with pytest.raises(SQLAlchemyError, match="missing greenlet"):
        await registry.get_model_async(object(), "panel_users")  # type: ignore[arg-type]


def test_registry_config_rejects_invalid_default_module_label():
    with pytest.raises(ValueError, match="valid dotted Python module path"):
        RegistryConfig(default_module_label="bad-module/path")


def test_registry_rejects_invalid_direct_module_label_override():
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    Table(
        "module_label_users",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String),
    )
    metadata.create_all(engine)

    registry = SqlModelRegistry()

    with pytest.raises(ValueError, match="valid dotted Python module path"):
        registry.get_model(engine, "module_label_users", module_label="bad-module/path")


def test_registry_rejects_module_label_outside_trusted_root():
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    Table(
        "external_module_users",
        metadata,
        Column("id", Integer, primary_key=True),
    )
    metadata.create_all(engine)

    registry = SqlModelRegistry()

    with pytest.raises(ValueError, match="trusted 'boti_data' module namespace"):
        registry.get_model(engine, "external_module_users", module_label="os.path")


def test_registry_rejects_existing_non_dynamic_module_namespace():
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    Table(
        "existing_module_users",
        metadata,
        Column("id", Integer, primary_key=True),
    )
    metadata.create_all(engine)

    registry = SqlModelRegistry()

    with pytest.raises(ValueError, match="unused or registry-managed dynamic module namespace"):
        registry.get_model(engine, "existing_module_users", module_label="boti_data")
