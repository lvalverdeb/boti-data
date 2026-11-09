"""Case 6: streaming a 1M-row result in Arrow record batches.

TODO(R7): switch to ``boti_data.db.arrow_fetch`` ``iter_batches()`` once it lands;
until then this is a batched cursor stand-in with the same shape (one Arrow
batch per 100k rows, never the whole result in memory).
"""

import pyarrow as pa
import pytest
from sqlalchemy import create_engine

from benchmarks._seed import ROWS_1M

BATCH_ROWS = 100_000
COLUMNS = ["id", "event_date", "group_id", "amount", "payload"]


def _stream_batches(engine) -> int:
    rows = 0
    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True).exec_driver_sql(
            "select id, event_date, group_id, amount, payload from events"
        )
        while chunk := result.fetchmany(BATCH_ROWS):
            batch = pa.record_batch(list(map(list, zip(*chunk))), names=COLUMNS)
            rows += batch.num_rows
    return rows


@pytest.mark.perf
def test_stream_1m_rows_in_arrow_batches(benchmark, sqlite_1m_dsn) -> None:
    engine = create_engine(sqlite_1m_dsn)
    try:
        rows = benchmark(lambda: _stream_batches(engine))
    finally:
        engine.dispose()
    assert rows == ROWS_1M
