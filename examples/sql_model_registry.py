"""
Dynamic ORM reflection example using SqlModelRegistry.
"""

from __future__ import annotations

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine

from boti_data.db import get_global_registry


def main() -> None:
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    Table(
        "panel_users",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("username", String, nullable=False),
    )
    metadata.create_all(engine)

    registry = get_global_registry()
    user_model = registry.get_model(engine, "panel_users")
    cached_model = registry.get_model(engine, "panel_users")

    print(user_model)
    print(f"tablename={user_model.__tablename__}")
    print(f"cache_hit={user_model is cached_model}")


if __name__ == "__main__":
    main()
