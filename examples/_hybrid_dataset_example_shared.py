"""Shared HybridDataset construction for data_hybrid_dataset.py / data_hybrid_dataset_distributed.py."""

from __future__ import annotations

from pathlib import Path

from boti_data import DataHelper, HybridDataset

# Shared by both callers so they can set/restore the same env var around _build_hybrid_dataset().
WORKER_DSN_ENV_VAR = "BOTI_EXAMPLE_HYBRID_SQLITE_DSN"


def _build_hybrid_dataset(db_path: Path) -> HybridDataset:
    sqlite_dsn = f"sqlite:///{db_path}"
    historical = DataHelper(
        backend="sqlalchemy",
        connection_url=sqlite_dsn,
        worker_connection_env_var=WORKER_DSN_ENV_VAR,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="historical_events",
    )
    live = DataHelper(
        backend="sqlalchemy",
        connection_url=sqlite_dsn,
        worker_connection_env_var=WORKER_DSN_ENV_VAR,
        poolclass="sqlalchemy.pool.NullPool",
        query_only=False,
        table="live_events",
    )
    return HybridDataset(historical, live, date_field="event_date", split_date="2026-04-18")
