"""Distributed HybridDataset example with a shared Dask session."""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from dask.distributed import LocalCluster
from sqlalchemy import Date, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataHelper, HybridDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hybrid_dataset_example_shared import _build_hybrid_dataset  # noqa: E402


class Base(DeclarativeBase):
    pass


class HistoricalEvent(Base):
    __tablename__ = "historical_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(16))


class LiveEvent(Base):
    __tablename__ = "live_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[dt.date] = mapped_column(Date())
    status: Mapped[str] = mapped_column(String(16))


def _seed_events(db_path: Path) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    HistoricalEvent(id=1, event_date=dt.date(2026, 4, 14), status="hist"),
                    HistoricalEvent(id=2, event_date=dt.date(2026, 4, 16), status="hist"),
                    LiveEvent(id=10, event_date=dt.date(2026, 4, 18), status="live"),
                    LiveEvent(id=11, event_date=dt.date(2026, 4, 19), status="live"),
                ]
            )
            session.commit()
    finally:
        engine.dispose()


def _run_distributed_hybrid_load(
    dataset: HybridDataset, cluster: LocalCluster
) -> dict[str, object]:
    with DataHelper.session(
        scheduler_address=cluster.scheduler_address,
        verify_connectivity=True,
        shared=True,
        shared_key="hybrid-dataset-distributed",
    ) as client:
        mixed = dataset.dask.load(start="2026-04-14", end="2026-04-19", diagnostics=True)
        async_rows = asyncio.run(
            dataset.aload(start="2026-04-16", end="2026-04-18", return_type="pandas")
        )
        computed = mixed.compute().sort_values("id").reset_index(drop=True)
        workers = len(client.scheduler_info()["workers"])

    return {
        "workers": workers,
        "sync_type": type(mixed).__name__,
        "sync_rows": int(len(computed)),
        "async_rows": int(len(async_rows)),
        "ids": computed["id"].tolist(),
        "statuses": computed["status"].tolist(),
    }


def run_example() -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "hybrid_distributed.db"
        _seed_events(db_path)

        dataset = _build_hybrid_dataset(db_path)

        with LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            dashboard_address=":0",
        ) as cluster:
            result = _run_distributed_hybrid_load(dataset, cluster)

        dataset.close()

    return result


def main() -> dict[str, object]:
    result = run_example()
    print(f"Hybrid distributed workers: {result['workers']}")
    print(f"Hybrid distributed sync type={result['sync_type']} rows={result['sync_rows']}")
    print(f"Hybrid distributed async rows={result['async_rows']}")
    print(f"Hybrid distributed ids={result['ids']}")
    return result


if __name__ == "__main__":
    main()
