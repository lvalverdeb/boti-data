"""
DataHelper worker_connection_env_var example.

Demonstrates the pickleable configuration pattern for distributed Dask workloads.

When Dask serialises tasks to send them to workers it pickles every argument,
including the config.  Embedding a live connection URL in a pickled payload has
two problems:

  1. SQLAlchemy engine and pool objects are not picklable and will raise an error.
  2. Even if the URL is a plain string, credentials travel through scheduler memory
     and may appear in logs.

Setting worker_connection_env_var on SqlDatabaseConfig tells boti-data to strip
the connection URL out of the worker payload.  Workers reconstruct their own
engine by reading the DSN from the named environment variable instead.

This example shows:
  - How to configure worker_connection_env_var
  - That the derived WorkerSqlConfig carries no connection_url
  - That workers can still resolve the DSN at runtime from the environment
  - A full distributed load using the safe config pattern
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from dask.distributed import LocalCluster
from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataHelper, SqlDatabaseConfig
from boti_data.db.sql_config import WorkerSqlConfig


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(32))
    amount: Mapped[float] = mapped_column()


def run_example() -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "worker_config_demo.db"

        bootstrap = create_engine(f"sqlite:///{db_path}")
        try:
            Base.metadata.create_all(bootstrap)
            with Session(bootstrap) as session:
                session.add_all(
                    [
                        Transaction(status="confirmed", amount=100.0),
                        Transaction(status="confirmed", amount=250.0),
                        Transaction(status="pending", amount=75.0),
                    ]
                )
                session.commit()
        finally:
            bootstrap.dispose()

        dsn = f"sqlite:///file:{db_path}?mode=ro&uri=true"
        env_var = "BOTI_EXAMPLE_WORKER_DSN"

        # ── Scheduler-side config ──────────────────────────────────────────
        # worker_connection_env_var names the env var that workers will read.
        # The full DSN lives only in this process; it is not pickled.
        scheduler_config = SqlDatabaseConfig(
            connection_url=dsn,
            poolclass="sqlalchemy.pool.NullPool",
            query_only=True,
            worker_connection_env_var=env_var,
        )

        # Demonstrate that WorkerSqlConfig strips the DSN.
        worker_payload = WorkerSqlConfig.from_database_config(scheduler_config)
        assert worker_payload.connection_url is None, "DSN must not be in the worker payload"
        assert worker_payload.connection_env_var == env_var

        # ── Worker-side environment ────────────────────────────────────────
        # In production this is injected via Kubernetes secrets or cluster
        # init scripts.  Here we set it in-process so that the LocalCluster
        # workers inherit it through forked memory.
        os.environ[env_var] = dsn

        try:
            with LocalCluster(
                n_workers=2,
                threads_per_worker=1,
                processes=False,  # threads share env; avoids subprocess injection
                dashboard_address=":0",
            ) as cluster:
                with DataHelper.session(scheduler_address=cluster.scheduler_address):
                    with DataHelper(scheduler_config, table="transactions") as helper:
                        ddf = helper.dask.load(status__exact="confirmed")
                        result = ddf.compute()
        finally:
            os.environ.pop(env_var, None)

    return {
        "worker_payload_has_no_dsn": worker_payload.connection_url is None,
        "worker_env_var": worker_payload.connection_env_var,
        "confirmed_rows": len(result),
        "total_amount": float(result["amount"].sum()),
    }


def main() -> dict[str, object]:
    result = run_example()
    print(f"Worker payload carries DSN:  {not result['worker_payload_has_no_dsn']}")
    print(f"Worker reads DSN from:       {result['worker_env_var']}")
    print(f"Confirmed transactions:      {result['confirmed_rows']}")
    print(f"Total confirmed amount:      {result['total_amount']}")
    return result


if __name__ == "__main__":
    main()
