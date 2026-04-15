"""
SqlPartitionedLoader example for lazy SQL -> Dask loading.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data.db import SqlDatabaseConfig, SqlPartitionedLoader


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str]
    description: Mapped[str]


def main() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "users.db"
        bootstrap_engine = create_engine(f"sqlite:///{db_path}")
        try:
            Base.metadata.create_all(bootstrap_engine)
            with Session(bootstrap_engine) as session:
                session.add_all(
                    [
                        User(id=1, status="active", description="urgent order"),
                        User(id=2, status="active", description="backlog item"),
                        User(id=3, status="inactive", description="routine followup"),
                        User(id=4, status="active", description="new escalation"),
                    ]
                )
                session.commit()
        finally:
            bootstrap_engine.dispose()

        config = SqlDatabaseConfig(
            connection_url=f"sqlite:///{db_path}",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
        )

        with SqlPartitionedLoader(config) as loader:
            frame = loader.load(
                statement=select(User),
                model=User,
                chunk_size=2,
            )

            print(f"Lazy partitions: {frame.npartitions}")
            print(frame.compute())


if __name__ == "__main__":
    main()
