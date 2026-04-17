"""
Dynamic ORM reflection example using SqlAlchemyModelBuilder.
"""

from __future__ import annotations

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine

from boti_data.db import BuilderConfig, SqlAlchemyModelBuilder


def main() -> None:
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    Table(
        "report_rows",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("status", String, nullable=False),
    )
    metadata.create_all(engine)

    builder = SqlAlchemyModelBuilder(
        engine,
        "report_rows",
        config=BuilderConfig(module_label="boti_data.examples.dynamic_models"),
    )
    model = builder.build_model()

    print(model)
    print(f"module={model.__module__}")
    print(f"columns={[column.name for column in model.__table__.columns]}")


if __name__ == "__main__":
    main()
