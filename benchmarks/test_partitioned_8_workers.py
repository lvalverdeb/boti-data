"""Case 5: a partitioned lazy load computed on an 8-worker local cluster."""

import pytest

from benchmarks._seed import ROWS_1M

PARTITIONS = 8


@pytest.mark.perf
def test_partitioned_load_on_8_workers(benchmark, sql_helper) -> None:
    pytest.importorskip("dask.distributed")
    from boti_data import DataHelper

    with DataHelper.session(
        cluster_kwargs={
            "n_workers": PARTITIONS,
            "threads_per_worker": 1,
            "processes": False,
            "dashboard_address": ":0",
        }
    ):

        def _load() -> int:
            frame = sql_helper.dask.load(chunk_size=ROWS_1M // PARTITIONS)
            return len(frame.compute())

        rows = benchmark(_load)
    assert rows == ROWS_1M
