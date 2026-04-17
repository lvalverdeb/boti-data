"""
DataHelper semi_join example.

Demonstrates two patterns from the README:

1. Semi-join with a pandas Series — load only rows whose key appears in a
   locally-held list of values.

2. Semi-join with a lazy Dask Series — both datasets remain as Dask graphs
   until a single .compute() triggers everything together.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataHelper, SqlDatabaseConfig


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    active: Mapped[int] = mapped_column()  # 1 = active, 0 = inactive
    name: Mapped[str] = mapped_column(String(64))


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column()
    amount: Mapped[float] = mapped_column()


def run_example() -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "semi_join_demo.db"
        readonly_dsn = f"sqlite:///file:{db_path}?mode=ro&uri=true"

        bootstrap = create_engine(f"sqlite:///{db_path}")
        try:
            Base.metadata.create_all(bootstrap)
            with Session(bootstrap) as session:
                session.add_all(
                    [
                        Customer(id=1, active=1, name="Alice"),
                        Customer(id=2, active=0, name="Bob"),
                        Customer(id=3, active=1, name="Carol"),
                        Customer(id=4, active=1, name="Dave"),
                    ]
                )
                session.add_all(
                    [
                        Order(id=1, customer_id=1, amount=100.0),
                        Order(id=2, customer_id=2, amount=200.0),
                        Order(id=3, customer_id=3, amount=150.0),
                        Order(id=4, customer_id=4, amount=300.0),
                    ]
                )
                session.commit()
        finally:
            bootstrap.dispose()

        config = SqlDatabaseConfig(
            connection_url=readonly_dsn,
            poolclass="sqlalchemy.pool.NullPool",
            query_only=True,
        )

        # ── Pattern 1: semi_join with a pandas Series ──────────────────────
        # The Series is a locally-held list of key values.  semi_join loads
        # only the rows whose key column appears in that Series.
        active_customer_ids = pd.Series([1, 3, 4], dtype="Int64")

        with DataHelper(config, table="orders") as helper:
            ddf = helper.semi_join(active_customer_ids, on="customer_id")
            pandas_result = ddf.compute().sort_values("id").reset_index(drop=True)

        # ── Pattern 2: semi_join with a lazy Dask Series ───────────────────
        # Both loaders stay lazy.  A single .compute() triggers both loads.
        with DataHelper(config, table="customers") as customer_helper:
            with DataHelper(config, table="orders") as order_helper:
                # Lazy Dask Series — no DB hit yet
                active_ids_ddf = customer_helper.dask.load(active__exact=1)["id"]

                # Attach the series as the semi-join key — still lazy
                orders_ddf = order_helper.semi_join(active_ids_ddf, on="customer_id")

                # Single compute() resolves both loads
                lazy_result = orders_ddf.compute().sort_values("id").reset_index(drop=True)

    return {
        "pandas_series_matched": len(pandas_result),
        "lazy_dask_series_matched": len(lazy_result),
        "pandas_customer_ids": sorted(pandas_result["customer_id"].tolist()),
        "lazy_customer_ids": sorted(lazy_result["customer_id"].tolist()),
    }


def main() -> dict[str, object]:
    result = run_example()
    print("Semi-join with pandas Series:")
    print(f"  matched rows: {result['pandas_series_matched']}")
    print(f"  customer_ids: {result['pandas_customer_ids']}")
    print()
    print("Semi-join with lazy Dask Series:")
    print(f"  matched rows: {result['lazy_dask_series_matched']}")
    print(f"  customer_ids: {result['lazy_customer_ids']}")
    return result


if __name__ == "__main__":
    main()
