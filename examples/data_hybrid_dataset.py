"""HybridDataset example: mixed windows, explicit sources, and engine-bound views."""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import Date, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import HybridDataset

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


def _seed(engine) -> None:
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                HistoricalEvent(id=1, event_date=dt.date(2026, 4, 14), status="hist"),
                HistoricalEvent(id=2, event_date=dt.date(2026, 4, 16), status="hist"),
                LiveEvent(id=10, event_date=dt.date(2026, 4, 18), status="live"),
                LiveEvent(id=11, event_date=dt.date(2026, 4, 19), status="live"),
                LiveEvent(id=12, event_date=dt.date(2026, 4, 20), status="live"),
            ]
        )
        session.commit()


def _load_hybrid_views(dataset: HybridDataset) -> dict[str, object]:
    sync_frame = dataset.load(start="2026-04-14", end="2026-04-20", return_type="auto")
    historical_frame = dataset.pandas.load(
        start="2026-04-14",
        end="2026-04-17",
        source="historical",
    )
    live_frame = dataset.polars.load(
        start="2026-04-18",
        end="2026-04-20",
        source="live",
    )
    async_frame = asyncio.run(
        dataset.aload(start="2026-04-16", end="2026-04-18", return_type="pandas")
    )
    return {
        "sync_type": type(sync_frame).__name__,
        "sync_rows": int(sync_frame.shape[0].compute()),
        "historical_type": type(historical_frame).__name__,
        "historical_rows": int(len(historical_frame)),
        "live_type": type(live_frame).__name__,
        "live_rows": int(live_frame.height),
        "async_type": type(async_frame).__name__,
        "async_rows": int(len(async_frame)),
    }


def _print_hybrid_result(result: dict[str, object]) -> None:
    print(f"Hybrid sync type={result['sync_type']} rows={result['sync_rows']}")
    print(
        f"Hybrid historical view type={result['historical_type']} rows={result['historical_rows']}"
    )
    print(f"Hybrid live view type={result['live_type']} rows={result['live_rows']}")
    print(f"Hybrid async type={result['async_type']} rows={result['async_rows']}")


def main() -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "hybrid_example.db"
        engine = create_engine(f"sqlite:///{db_path}")
        try:
            _seed(engine)
        finally:
            engine.dispose()

        dataset = _build_hybrid_dataset(db_path)

        try:
            result = _load_hybrid_views(dataset)
        finally:
            dataset.close()

    _print_hybrid_result(result)
    return result


if __name__ == "__main__":
    main()
