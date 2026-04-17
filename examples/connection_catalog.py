"""
ConnectionCatalog example — named SQL profile registration and lookup.

Demonstrates how to register SQL configs under named keys so that connection
details are managed centrally rather than scattered across helpers.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import ConnectionCatalog, DataHelper, SqlDatabaseConfig


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(32))
    region: Mapped[str] = mapped_column(String(32))


def run_example() -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "catalog_demo.db"
        readonly_dsn = f"sqlite:///file:{db_path}?mode=ro&uri=true"

        # Bootstrap the database with seed data.
        bootstrap = create_engine(f"sqlite:///{db_path}")
        try:
            Base.metadata.create_all(bootstrap)
            with Session(bootstrap) as session:
                session.add_all(
                    [
                        Order(status="confirmed", region="EMEA"),
                        Order(status="confirmed", region="AMER"),
                        Order(status="pending", region="EMEA"),
                        Order(status="confirmed", region="APAC"),
                    ]
                )
                session.commit()
        finally:
            bootstrap.dispose()

        # Register the connection under a named key in the catalog.
        # In production this would typically be driven by environment variables
        # via catalog.load_sql("prod", prefix="PROD_DB_").
        catalog = ConnectionCatalog()
        catalog.register_sql(
            "orders_db",
            SqlDatabaseConfig(
                connection_url=readonly_dsn,
                poolclass="sqlalchemy.pool.NullPool",
                query_only=True,
            ),
        )

        # Retrieve the named config and wire it into a DataHelper.
        orders_config = catalog.sql_config("orders_db")

        with DataHelper(orders_config, table="orders") as helper:
            # Pandas — eager, filter applied at load time
            all_confirmed = helper.pandas.load(status__exact="confirmed")

            # Polars — same query, different engine
            emea_confirmed = helper.polars.load(
                status__exact="confirmed",
                region__exact="EMEA",
            )

            # Dask — lazy; computation deferred until .compute()
            ddf = helper.dask.load(status__exact="confirmed")
            lazy_count = int(ddf["id"].count().compute())

    return {
        "confirmed_rows": len(all_confirmed),
        "emea_confirmed_rows": len(emea_confirmed),
        "lazy_confirmed_count": lazy_count,
    }


def main() -> dict[str, object]:
    result = run_example()
    print(f"All confirmed orders (pandas): {result['confirmed_rows']}")
    print(f"EMEA confirmed orders (polars): {result['emea_confirmed_rows']}")
    print(f"Confirmed orders via lazy Dask count: {result['lazy_confirmed_count']}")
    return result


if __name__ == "__main__":
    main()
