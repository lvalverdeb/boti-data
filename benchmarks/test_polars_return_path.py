"""Case 3: the polars return path over the same 1M-row SQL source (R13 target)."""

import pytest

from benchmarks._seed import ROWS_1M


@pytest.mark.perf
def test_sql_load_1m_rows_to_polars(benchmark, sql_helper) -> None:
    rows = benchmark(lambda: sql_helper.polars.load().height)
    assert rows == ROWS_1M
