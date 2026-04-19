"""HybridDataset example combining parquet historical data with SQL live data."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sqlalchemy import Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataHelper, HybridDataset


class Base(DeclarativeBase):
    pass


class LiveOrder(Base):
    __tablename__ = "live_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_date: Mapped[str] = mapped_column(String(10))
    amount: Mapped[int] = mapped_column(Integer)


def main() -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        parquet_root = tmp_path / "historical"
        parquet_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "id": [1, 2],
                "order_date": ["2026-04-15", "2026-04-17"],
                "amount": [100, 150],
            }
        ).to_parquet(parquet_root / "orders.parquet", index=False)

        db_path = tmp_path / "live_orders.db"
        engine = create_engine(f"sqlite:///{db_path}")
        try:
            Base.metadata.create_all(engine)
            with Session(engine) as session:
                session.add_all(
                    [
                        LiveOrder(id=10, order_date="2026-04-18", amount=200),
                        LiveOrder(id=11, order_date="2026-04-19", amount=250),
                    ]
                )
                session.commit()
        finally:
            engine.dispose()

        historical = DataHelper(
            backend="parquet",
            storage_path=str(parquet_root),
            parquet_filename="orders",
        )
        live = DataHelper(
            backend="sqlalchemy",
            connection_url=f"sqlite:///{db_path}",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
            table="live_orders",
        )

        dataset = HybridDataset(
            historical,
            live,
            date_field="order_date",
            split_date="2026-04-18",
        )
        try:
            mixed = dataset.load(start="2026-04-15", end="2026-04-19", return_type="dask")
            eager = asyncio.run(
                dataset.aload(start="2026-04-15", end="2026-04-19", return_type="pandas")
            )
            mixed_rows = int(mixed.shape[0].compute())
            eager_rows = int(len(eager))
            total_amount = int(eager["amount"].sum())
        finally:
            dataset.close()

    result = {
        "mixed_type": type(mixed).__name__,
        "mixed_rows": mixed_rows,
        "eager_type": type(eager).__name__,
        "eager_rows": eager_rows,
        "total_amount": total_amount,
    }
    print(f"Hybrid parquet+sql mixed type={result['mixed_type']} rows={result['mixed_rows']}")
    print(f"Hybrid parquet+sql eager type={result['eager_type']} rows={result['eager_rows']}")
    print(f"Hybrid parquet+sql total amount={result['total_amount']}")
    return result


if __name__ == "__main__":
    main()

