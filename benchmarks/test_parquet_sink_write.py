"""Case 7: writing a 100k-row frame through ParquetSink, partitioned by date."""

import pytest

from benchmarks._seed import ROWS_100K, event_frame


@pytest.mark.perf
def test_parquet_sink_write_100k_rows(benchmark, bench_root) -> None:
    from boti_data import write_parquet

    frame = event_frame(1, ROWS_100K + 1).rename(columns={"event_date": "partition_date"})
    destination = {
        "storage_path": str(bench_root / "sink_out"),
        "project_root": str(bench_root),
        "partition_on": ["partition_date"],
    }
    result = benchmark(lambda: write_parquet(destination, frame, overwrite=True))
    assert result.files
