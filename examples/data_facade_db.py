"""
DataGateway example for SQL-backed loading.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data.db import SqlDatabaseConfig
from boti_data.gateway import DataGateway


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str]
    description: Mapped[str]


def _seed_users_db(db_path: Path) -> None:
    bootstrap_engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(bootstrap_engine)
        with Session(bootstrap_engine) as session:
            session.add_all(
                [
                    User(status="active", description="urgent order"),
                    User(status="inactive", description="routine followup"),
                ]
            )
            session.commit()
    finally:
        bootstrap_engine.dispose()


def _load_all_frames(facade: DataGateway) -> dict[str, object]:
    active = {"status__exact": "active"}
    return {
        "dask": facade.load(statement=select(User), model=User, filters=active),
        "pandas": facade.load(
            statement=select(User), model=User, filters=active, return_type="pandas"
        ),
        "arrow": facade.load(
            statement=select(User), model=User, filters=active, return_type="arrow"
        ),
        "polars": facade.load(
            statement=select(User), model=User, filters=active, return_type="polars"
        ),
        "auto": facade.load(statement=select(User), model=User, filters=active, return_type="auto"),
        "lazy_pandas": facade.load(
            statement=select(User),
            model=User,
            filters=active,
            return_type="pandas",
            execution_mode="lazy",
        ),
        "auto_chunked": facade.load(
            statement=select(User),
            model=User,
            filters={"id__in": [1, 2]},
            as_pandas=True,
            in_chunk_strategy="auto",
        ),
    }


def _print_frames(frames: dict[str, object]) -> None:
    print("dask")
    print(frames["dask"].compute())
    print("\npandas")
    print(frames["pandas"])
    print("\narrow")
    print(frames["arrow"])
    print("\npolars")
    print(frames["polars"])
    print("\nauto")
    print(frames["auto"])
    print("\npandas over lazy fetch")
    print(frames["lazy_pandas"])
    print("\nauto IN-chunking policy")
    print(frames["auto_chunked"])


def main() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "users.db"
        _seed_users_db(db_path)

        config = SqlDatabaseConfig(
            connection_url=f"sqlite:///{db_path}",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
        )

        with DataGateway(config) as facade:
            frames = _load_all_frames(facade)
            _print_frames(frames)


if __name__ == "__main__":
    main()
