"""
Unit test suite verifying SqlAlchemyModelBuilder bindings.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine

from boti_data.db.sql_model_builder import BuilderConfig, SqlAlchemyModelBuilder
from boti_data.db.sql_model_registry import get_global_registry


def test_sql_model_builder_resolves_registry_transparently() -> None:
    engine = create_engine("sqlite:///:memory:")

    metadata = MetaData()
    test_table = Table(
        "facade_users", metadata, Column("id", Integer, primary_key=True), Column("name", String)
    )
    metadata.create_all(engine)

    builder = SqlAlchemyModelBuilder(engine, "facade_users")
    FacadeModel = builder.build_model()

    assert FacadeModel.__name__ == "FacadeUsers"
    assert FacadeModel.__tablename__ == "facade_users"

    # Query registry back manually to assert singleton cache hit
    RegistryModel = get_global_registry().get_model(engine, "facade_users")
    assert FacadeModel is RegistryModel


def test_sql_model_builder_string_normalizations() -> None:
    assert SqlAlchemyModelBuilder._normalize_class_name("super_admin_nodes") == "SuperAdminNodes"

    # Assert column normalization sanitizes reserved keywords securely
    assert SqlAlchemyModelBuilder._normalize_column_name("lambda") == "lambda_field"
    assert SqlAlchemyModelBuilder._normalize_column_name("1invalid_start") == "_1invalid_start"


def test_builder_config_rejects_invalid_module_label() -> None:
    with pytest.raises(ValueError, match="valid dotted Python module path"):
        BuilderConfig(module_label="bad-module/path")
