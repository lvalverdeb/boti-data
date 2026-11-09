"""
ADBC partition fetches: Arrow straight from the driver, only behind the read-only guards.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from boti_data.db import adbc_fetch
from boti_data.db.partitioned_execution import SqlPartitionExecutor
from boti_data.db.partitioned_types import SqlPartitionSpec
from boti_data.db.sql_config import WorkerSqlConfig

ROWS = 200_000
META = {"id": "Int64", "group_id": "Int64", "amount": "Float64", "label": "string"}
PARTITION = SqlPartitionSpec(
    sql="select id, group_id, amount, label from events where id >= ? order by id", params=(1,)
)


def seed(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute(
        "create table events (id integer primary key, group_id integer, amount real, label text)"
    )
    con.executemany(
        "insert into events values (?, ?, ?, ?)",
        ((i, i % 97, i * 0.25, f"event-{i}") for i in range(1, ROWS + 1)),
    )
    con.commit()
    con.close()


def read_only_dsn(path: Path) -> str:
    # query_only=True on SQLite needs the file URI form with mode=ro; the same
    # URI is what the ADBC driver opens, so the file is read-only end to end.
    return f"sqlite:///file:{path}?mode=ro&uri=true"


def show_eligibility(dsn: str) -> dict[str, bool]:
    """ADBC is used only when all three hold: read-only, driver installed, not opted out."""
    cases = {
        "query_only=True (default)": WorkerSqlConfig(connection_url=dsn, query_only=True),
        "query_only=False": WorkerSqlConfig(connection_url=dsn, query_only=False),
        "adbc=False": WorkerSqlConfig(connection_url=dsn, query_only=True, adbc=False),
        "mysql (no ADBC driver)": WorkerSqlConfig(
            connection_url="mysql+pymysql://u:p@db/app", query_only=True
        ),
    }
    verdicts = {name: adbc_fetch.adbc_eligible(config) for name, config in cases.items()}
    print("Eligibility:")
    for name, verdict in verdicts.items():
        print(f"  {name:28s} -> {'ADBC' if verdict else 'SQLAlchemy'}")
    return verdicts


def show_read_only_session(dsn: str) -> str:
    """The ADBC session gets the same read-only statement the SQLAlchemy engine issues."""
    config = WorkerSqlConfig(connection_url=dsn, query_only=True)
    with adbc_fetch.open_read_only(config) as connection, connection.cursor() as cursor:
        try:
            cursor.execute("insert into events values (0, 0, 0.0, 'x')")
        except Exception as exc:  # the driver refuses: the file is mode=ro and PRAGMA query_only=1
            message = f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}"
    print(f"Write through the ADBC session refused -> {message}")
    return message


def timed_fetch(dsn: str, *, adbc: bool) -> tuple[int, float]:
    config = WorkerSqlConfig(connection_url=dsn, query_only=True, adbc=adbc)
    started = perf_counter()
    frame = SqlPartitionExecutor.fetch_partition(
        config=config,
        gate_key="example",
        max_concurrent_fetches=1,
        partition=PARTITION,
        meta_dtypes=META,
        use_arrow=True,
    )
    return len(frame), perf_counter() - started


def main() -> dict[str, Any]:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "events.sqlite"
        seed(db_path)
        dsn = read_only_dsn(db_path)
        installed = adbc_fetch.driver_installed("sqlite")
        print(f"ADBC SQLite driver installed: {installed} (pip install 'boti-data[adbc]')")
        summary: dict[str, Any] = {
            "driver_installed": installed,
            "eligibility": show_eligibility(dsn),
        }
        if not installed:
            print("Without the driver every fetch takes the SQLAlchemy Arrow path.")
            return summary
        summary["refusal"] = show_read_only_session(dsn)
        rows_sa, seconds_sa = timed_fetch(dsn, adbc=False)
        rows_adbc, seconds_adbc = timed_fetch(dsn, adbc=True)
        print(f"{ROWS:,} rows, SQLAlchemy Arrow path: {seconds_sa:.3f}s")
        print(
            f"{ROWS:,} rows, ADBC path:            {seconds_adbc:.3f}s  ({seconds_sa / seconds_adbc:.1f}x)"
        )
        summary.update(rows=(rows_sa, rows_adbc), seconds=(seconds_sa, seconds_adbc))
        return summary


if __name__ == "__main__":
    main()
