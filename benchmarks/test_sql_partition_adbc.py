"""Case 9: one 1M-row read-only partition fetch, ADBC versus the SQLAlchemy Arrow path."""

from __future__ import annotations

import pytest

from benchmarks._seed import ROWS_1M
from boti_data.db.partitioned_execution import SqlPartitionExecutor
from boti_data.db.partitioned_types import SqlPartitionSpec
from boti_data.db.sql_config import WorkerSqlConfig

META = {
    "id": "Int64",
    "event_date": "string",
    "group_id": "Int64",
    "amount": "Float64",
    "payload": "string",
}
SQL = "select id, event_date, group_id, amount, payload from events where id >= ? order by id"


def _read_only(dsn: str) -> str:
    return dsn.replace("sqlite:///", "sqlite:///file:") + "?mode=ro&uri=true"


@pytest.mark.perf
@pytest.mark.parametrize("adbc", [False, True], ids=["sqlalchemy-arrow", "adbc"])
def test_partition_fetch_1m_rows(benchmark, sqlite_1m_dsn, adbc) -> None:
    if adbc:
        pytest.importorskip("adbc_driver_sqlite")
    config = WorkerSqlConfig(connection_url=_read_only(sqlite_1m_dsn), query_only=True, adbc=adbc)

    def _fetch() -> int:
        return len(
            SqlPartitionExecutor.fetch_partition(
                config=config,
                gate_key="bench",
                max_concurrent_fetches=1,
                partition=SqlPartitionSpec(sql=SQL, params=(1,)),
                meta_dtypes=META,
                use_arrow=True,
            )
        )

    assert benchmark(_fetch) == ROWS_1M
