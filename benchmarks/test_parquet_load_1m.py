"""Case 2: 1M rows from a hive-partitioned Parquet dataset into pandas."""

import pytest

from benchmarks._seed import ROWS_1M


@pytest.mark.perf
def test_parquet_load_1m_rows_to_pandas(benchmark, parquet_1m_root, bench_root) -> None:
    from boti_data import DataHelper

    helper = DataHelper(
        backend="parquet",
        storage_path=str(parquet_1m_root),
        project_root=str(bench_root),
        parquet_start_date="2026-01-01",
        parquet_end_date="2026-12-31",
    )
    try:
        rows = benchmark(lambda: len(helper.pandas.load()))
    finally:
        helper.close()
    assert rows == ROWS_1M
