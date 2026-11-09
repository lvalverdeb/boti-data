"""Case 1: 1M rows from SQL into pandas through DataGateway (PRD success metric)."""

import pytest

from benchmarks._seed import ROWS_1M


@pytest.mark.perf
def test_sql_load_1m_rows_to_pandas(benchmark, sql_helper) -> None:
    rows = benchmark(lambda: len(sql_helper.pandas.load()))
    assert rows == ROWS_1M
