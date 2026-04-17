"""
DataGateway example for Dask distributed execution.
"""

from __future__ import annotations

import asyncio
import os
from time import perf_counter
from pathlib import Path
from tempfile import TemporaryDirectory

from dask.distributed import LocalCluster
from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data.db import SqlDatabaseConfig
from boti_dask import dask_session
from boti_data.gateway import DataGateway
from boti_data.joins import indexed_left_join


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str]
    description: Mapped[str]


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    tier: Mapped[str]
    region: Mapped[str]


LEFT_ROWS = int(os.environ.get("BOTI_EXAMPLE_LEFT_ROWS", "2000000"))
RIGHT_ROWS = int(os.environ.get("BOTI_EXAMPLE_RIGHT_ROWS", "1500000"))
BATCH_SIZE = int(os.environ.get("BOTI_EXAMPLE_BATCH_SIZE", "50000"))
ENABLE_DIAGNOSTICS = os.environ.get("BOTI_EXAMPLE_DIAGNOSTICS", "0") == "1"


def _seed_tables(db_path: Path) -> None:
    bootstrap_engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(bootstrap_engine)
        with Session(bootstrap_engine) as session:
            for start in range(1, LEFT_ROWS + 1, BATCH_SIZE):
                stop = min(start + BATCH_SIZE, LEFT_ROWS + 1)
                session.execute(
                    User.__table__.insert(),
                    [
                        {
                            "id": idx,
                            "status": "active" if idx % 5 else "inactive",
                            "description": f"order-{idx % 1000:04d}",
                        }
                        for idx in range(start, stop)
                    ],
                )
                session.commit()

            for start in range(1, RIGHT_ROWS + 1, BATCH_SIZE):
                stop = min(start + BATCH_SIZE, RIGHT_ROWS + 1)
                session.execute(
                    UserProfile.__table__.insert(),
                    [
                        {
                            "id": idx,
                            "tier": "gold" if idx % 10 == 0 else "standard",
                            "region": f"region-{idx % 12:02d}",
                        }
                        for idx in range(start, stop)
                    ],
                )
                session.commit()
    finally:
        bootstrap_engine.dispose()


def _summarize_run(*, users, profiles, joined, load_seconds: float, mode: str) -> None:
    print(f"\n{mode} load benchmark")
    print(f"load/setup seconds: {load_seconds:.2f}")
    print(f"User partitions: {users.npartitions}")
    print(f"Profile partitions: {profiles.npartitions}")
    print(f"Joined partitions: {joined.npartitions}")
    print("Join head:")
    print(joined.head(5))
    print("\nStatus counts:")
    print(users["status"].value_counts(split_out=1).compute().sort_index())
    print("\nJoin match summary:")
    matched = joined["tier"].count().compute()
    unmatched = joined["tier"].isna().sum().compute()
    print(f"matched rows: {matched:,}")
    print(f"unmatched rows: {unmatched:,}")


def _run_sync_benchmark(config: SqlDatabaseConfig) -> None:
    with DataGateway(config) as facade:
        started = perf_counter()
        users = facade.load(
            statement=select(User),
            model=User,
            chunk_size=50_000,
            persist=True,
            diagnostics=ENABLE_DIAGNOSTICS,
        )
        profiles = facade.load(
            statement=select(UserProfile),
            model=UserProfile,
            chunk_size=50_000,
            persist=True,
            diagnostics=ENABLE_DIAGNOSTICS,
        )
        joined = indexed_left_join(
            users,
            profiles,
            join_key="id",
            join_schema_map={"id": "Int64"},
            persist=True,
            diagnostics=ENABLE_DIAGNOSTICS,
        )
        load_seconds = perf_counter() - started
    _summarize_run(users=users, profiles=profiles, joined=joined, load_seconds=load_seconds, mode="sync")


async def _run_async_benchmark(config: SqlDatabaseConfig) -> None:
    async with DataGateway(config) as facade:
        started = perf_counter()
        users = await facade.aload(
            statement=select(User),
            model=User,
            chunk_size=50_000,
            persist=True,
            diagnostics=ENABLE_DIAGNOSTICS,
        )
        profiles = await facade.aload(
            statement=select(UserProfile),
            model=UserProfile,
            chunk_size=50_000,
            persist=True,
            diagnostics=ENABLE_DIAGNOSTICS,
        )
        joined = indexed_left_join(
            users,
            profiles,
            join_key="id",
            join_schema_map={"id": "Int64"},
            persist=True,
            diagnostics=ENABLE_DIAGNOSTICS,
        )
        load_seconds = perf_counter() - started
    _summarize_run(users=users, profiles=profiles, joined=joined, load_seconds=load_seconds, mode="async")


def main() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "users.db"
        _seed_tables(db_path)

        config = SqlDatabaseConfig(
            connection_url=f"sqlite:///{db_path}",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
        )

        with LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=False,
            dashboard_address=None,
        ) as cluster:
            print(f"Left rows: {LEFT_ROWS:,}")
            print(f"Right rows: {RIGHT_ROWS:,}")
            with dask_session(scheduler_address=cluster.scheduler_address):
                _run_sync_benchmark(config)
                asyncio.run(_run_async_benchmark(config))


if __name__ == "__main__":
    main()
