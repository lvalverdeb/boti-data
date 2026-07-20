"""Shared HybridDataset construction for data_hybrid_dataset.py / data_hybrid_dataset_distributed.py."""

from __future__ import annotations

from pathlib import Path

from boti_data import DataHelper, HybridDataset


def _build_hybrid_dataset(db_path: Path) -> HybridDataset:
    historical = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="historical_events",
    )
    live = DataHelper(
        backend="sqlalchemy",
        connection_url=f"sqlite:///{db_path}",
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="live_events",
    )
    return HybridDataset(historical, live, date_field="event_date", split_date="2026-04-18")
