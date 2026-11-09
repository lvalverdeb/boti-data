"""Case 4: an IN-clause filter over 1M rows (see docs/proposals on IN-clause response time)."""

import pytest

from benchmarks._seed import GROUPS, ROWS_1M

GROUP_IDS = list(range(0, GROUPS, 4))  # 25 of 100 groups -> a quarter of the rows


@pytest.mark.perf
def test_in_clause_filter_to_pandas(benchmark, sql_helper) -> None:
    rows = benchmark(lambda: len(sql_helper.pandas.load(group_id__in=GROUP_IDS)))
    assert rows == ROWS_1M // 4
