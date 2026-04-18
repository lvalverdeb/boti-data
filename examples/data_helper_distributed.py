"""
DataHelper example for distributed session and join convenience.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from dask.distributed import LocalCluster
from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataHelper


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str]


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    tier: Mapped[str]


def run_example() -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "helper_distributed.db"
        engine = create_engine(f"sqlite:///{db_path}")
        try:
            Base.metadata.create_all(engine)
            with Session(engine) as session:
                session.add_all(
                    [
                        User(id=1, status="active"),
                        User(id=2, status="inactive"),
                        User(id=3, status="active"),
                        User(id=4, status="active"),
                    ]
                )
                session.add_all(
                    [
                        UserProfile(id=1, tier="gold"),
                        UserProfile(id=3, tier="standard"),
                        UserProfile(id=4, tier="gold"),
                    ]
                )
                session.commit()
        finally:
            engine.dispose()

        config = {
            "backend": "sqlalchemy",
            "connection_url": f"sqlite:///{db_path}",
            "poolclass": "sqlalchemy.pool.NullPool",
            "query_only": False,
        }

        with LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=False,
            dashboard_address=":0",
        ) as cluster:
            with DataHelper.session(
                scheduler_address=cluster.scheduler_address,
                verify_connectivity=True,
                shared=True,
                shared_key="data-helper-distributed",
            ) as client:
                with DataHelper.session(
                    scheduler_address=cluster.scheduler_address,
                    verify_connectivity=True,
                    shared=True,
                    shared_key="data-helper-distributed",
                ) as shared_client:
                    workers = len(client.scheduler_info()["workers"])
                    shared_client_reused = client is shared_client
                    with DataHelper(config) as helper:
                        dry_run_users = helper.dask.load(
                            statement=select(User),
                            model=User,
                            dry_run=True,
                            diagnostics=True,
                        )
                        user_preview = helper.preview(
                            statement=select(User),
                            model=User,
                            n=2,
                        )
                        users = helper.load(statement=select(User), model=User)
                        profiles = helper.load(statement=select(UserProfile), model=UserProfile)
                        joined = DataHelper.left_join(
                            users,
                            profiles,
                            join_key="id",
                            join_schema_map={"id": "Int64"},
                            persist=True,
                        )
                        frame = joined.compute().sort_values("id").reset_index(drop=True)

    return {
        "workers": workers,
        "shared_client_reused": shared_client_reused,
        "dry_run_partitions": dry_run_users.npartitions,
        "preview_records": user_preview.to_dict(orient="records"),
        "matched_rows": int(frame["tier"].count()),
        "unmatched_rows": int(frame["tier"].isna().sum()),
        "records": frame.to_dict(orient="records"),
    }


def main() -> dict[str, object]:
    result = run_example()
    print(f"Distributed helper workers: {result['workers']}")
    print(f"Shared session reused: {result['shared_client_reused']}")
    print(f"Dry-run partitions: {result['dry_run_partitions']}")
    print("Helper preview records:")
    for record in result["preview_records"]:
        print(record)
    print("Joined helper records:")
    for record in result["records"]:
        print(record)
    print(f"matched rows: {result['matched_rows']}")
    print(f"unmatched rows: {result['unmatched_rows']}")
    return result


if __name__ == "__main__":
    main()
