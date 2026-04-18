"""
DataGateway example showing explicit Dask session management and diagnostics logs.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data.db import SqlDatabaseConfig
from boti_dask import dask_session, describe_client
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


def _example_settings() -> dict[str, int]:
    return {
        "user_rows": int(os.environ.get("BOTI_EXAMPLE_DIAGNOSTIC_LEFT_ROWS", "2500000")),
        "profile_rows": int(os.environ.get("BOTI_EXAMPLE_DIAGNOSTIC_RIGHT_ROWS", "2000000")),
        "batch_size": int(os.environ.get("BOTI_EXAMPLE_DIAGNOSTIC_BATCH_SIZE", "25000")),
        "workers": int(os.environ.get("BOTI_EXAMPLE_DIAGNOSTIC_WORKERS", "2")),
        "threads_per_worker": int(
            os.environ.get("BOTI_EXAMPLE_DIAGNOSTIC_THREADS_PER_WORKER", "1")
        ),
    }


def _seed_tables(db_path: Path, *, user_rows: int, profile_rows: int, batch_size: int) -> None:
    bootstrap_engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(bootstrap_engine)
        with Session(bootstrap_engine) as session:
            for start in range(1, user_rows + 1, batch_size):
                stop = min(start + batch_size, user_rows + 1)
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

            for start in range(1, profile_rows + 1, batch_size):
                stop = min(start + batch_size, profile_rows + 1)
                session.execute(
                    UserProfile.__table__.insert(),
                    [
                        {
                            "id": idx,
                            "tier": "gold" if idx % 10 == 0 else "standard",
                        }
                        for idx in range(start, stop)
                    ],
                )
                session.commit()
    finally:
        bootstrap_engine.dispose()


def run_example() -> dict[str, object]:
    settings = _example_settings()
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "diagnostics.db"
        _seed_tables(
            db_path,
            user_rows=settings["user_rows"],
            profile_rows=settings["profile_rows"],
            batch_size=settings["batch_size"],
        )

        config = SqlDatabaseConfig(
            connection_url=f"sqlite:///{db_path}",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
        )

        with dask_session(
            cluster_kwargs={
                "n_workers": settings["workers"],
                "threads_per_worker": settings["threads_per_worker"],
                "processes": False,
                "dashboard_address": ":0",
            }
        ) as client:
            client_summary = describe_client(client)
            with DataGateway(config) as facade:
                users = facade.load(
                    statement=select(User),
                    model=User,
                    persist=True,
                    diagnostics=True,
                )
                profiles = facade.load(
                    statement=select(UserProfile),
                    model=UserProfile,
                    persist=True,
                    diagnostics=True,
                )
                joined = indexed_left_join(
                    users,
                    profiles,
                    join_key="id",
                    join_schema_map={"id": "Int64"},
                    persist=True,
                    diagnostics=True,
                )
                head = joined.head(5)
                matched_rows = int(joined["tier"].count().compute())
                unmatched_rows = int(joined["tier"].isna().sum().compute())
                status_counts = (
                    users["status"].value_counts(split_out=1).compute().sort_index().to_dict()
                )

    return {
        "client": client_summary,
        "user_rows": settings["user_rows"],
        "profile_rows": settings["profile_rows"],
        "head": head,
        "user_partitions": users.npartitions,
        "profile_partitions": profiles.npartitions,
        "joined_partitions": joined.npartitions,
        "status_counts": status_counts,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
    }


def main() -> dict[str, object]:
    result = run_example()
    print("Managed session client:")
    print(result["client"])
    print(f"\nLeft rows: {result['user_rows']:,}")
    print(f"Right rows: {result['profile_rows']:,}")
    print(f"User partitions: {result['user_partitions']}")
    print(f"Profile partitions: {result['profile_partitions']}")
    print(f"Joined partitions: {result['joined_partitions']}")
    print("\nJoined head:")
    print(result["head"])
    print("\nStatus counts:")
    for status, count in result["status_counts"].items():
        print(f"{status}: {count:,}")
    print("\nMatch summary:")
    print(f"matched rows: {result['matched_rows']}")
    print(f"unmatched rows: {result['unmatched_rows']}")
    return result


if __name__ == "__main__":
    main()
