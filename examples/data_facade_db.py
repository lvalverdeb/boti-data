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


def main() -> None:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "users.db"
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

        config = SqlDatabaseConfig(
            connection_url=f"sqlite:///{db_path}",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
        )

        with DataGateway(config) as facade:
            dask_frame = facade.load(
                statement=select(User),
                model=User,
                filters={"status__exact": "active"},
            )
            pandas_frame = facade.load(
                statement=select(User),
                model=User,
                filters={"status__exact": "active"},
                return_type="pandas",
            )
            arrow_table = facade.load(
                statement=select(User),
                model=User,
                filters={"status__exact": "active"},
                return_type="arrow",
            )
            polars_frame = facade.load(
                statement=select(User),
                model=User,
                filters={"status__exact": "active"},
                return_type="polars",
            )
            auto_frame = facade.load(
                statement=select(User),
                model=User,
                filters={"status__exact": "active"},
                return_type="auto",
            )
            lazy_pandas_frame = facade.load(
                statement=select(User),
                model=User,
                filters={"status__exact": "active"},
                return_type="pandas",
                execution_mode="lazy",
            )
            auto_chunked_frame = facade.load(
                statement=select(User),
                model=User,
                filters={"id__in": [1, 2]},
                as_pandas=True,
                in_chunk_strategy="auto",
            )

            print("dask")
            print(dask_frame.compute())
            print("\npandas")
            print(pandas_frame)
            print("\narrow")
            print(arrow_table)
            print("\npolars")
            print(polars_frame)
            print("\nauto")
            print(auto_frame)
            print("\npandas over lazy fetch")
            print(lazy_pandas_frame)
            print("\nauto IN-chunking policy")
            print(auto_chunked_frame)


if __name__ == "__main__":
    main()
