from __future__ import annotations

import asyncio
import os
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from sqlalchemy import create_engine, text

from boti_data import DataHelper

ROW_COUNT = 200_000
IN_SIZES = (100, 1_000, 5_000, 10_000)
REPEATS = 3
SEED = 42


def _seed_sqlite(db_path: Path) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "create table events (id integer primary key, group_id integer not null, payload text not null)"
            )
            conn.exec_driver_sql("create index idx_events_group_id on events(group_id)")
            batch_size = 10_000
            for start in range(1, ROW_COUNT + 1, batch_size):
                stop = min(start + batch_size, ROW_COUNT + 1)
                rows = [
                    {
                        "id": i,
                        "group_id": i % 100,
                        "payload": f"payload-{i}",
                    }
                    for i in range(start, stop)
                ]
                conn.execute(
                    text(
                        "insert into events(id, group_id, payload) values (:id, :group_id, :payload)"
                    ),
                    rows,
                )
    finally:
        engine.dispose()


def _measure(fn: Callable[[], int]) -> tuple[float, int]:
    started = time.perf_counter()
    rows = fn()
    elapsed = (time.perf_counter() - started) * 1000.0
    return elapsed, rows


def _median_ms(samples: list[float]) -> float:
    return round(statistics.median(samples), 2)


def _build_scenarios(helper: DataHelper) -> dict[str, Callable[[list[int]], int]]:
    return {
        "sync_pandas_single": lambda ids: len(
            helper.pandas.load(id__in=ids, in_chunk_strategy="off")
        ),
        "sync_pandas_auto_chunk": lambda ids: len(helper.pandas.load(id__in=ids)),
        "sync_pandas_forced_chunk": lambda ids: len(
            helper.pandas.load(id__in=ids, in_chunk_size=900, in_chunk_concurrency=4)
        ),
        "sync_dask_single": lambda ids: int(
            helper.dask.load(id__in=ids, in_chunk_strategy="off").shape[0].compute()
        ),
        "sync_dask_auto_chunk": lambda ids: int(helper.dask.load(id__in=ids).shape[0].compute()),
        "async_pandas_single": lambda ids: len(
            asyncio.run(helper.pandas.aload(id__in=ids, in_chunk_strategy="off"))
        ),
        "async_pandas_auto_chunk": lambda ids: len(asyncio.run(helper.pandas.aload(id__in=ids))),
    }


def _benchmark_scenario(
    name: str, scenario: Callable[[list[int]], int], ids: list[int], in_size: int
) -> dict[str, Any]:
    warm_rows = scenario(ids)
    if warm_rows != in_size:
        raise RuntimeError(
            f"Warm-up mismatch for {name} in_size={in_size}: {warm_rows} != {in_size}"
        )

    samples: list[float] = []
    for _ in range(REPEATS):
        elapsed_ms, rows = _measure(lambda: scenario(ids))
        if rows != in_size:
            raise RuntimeError(f"Row mismatch for {name} in_size={in_size}: {rows} != {in_size}")
        samples.append(elapsed_ms)

    return {
        "scenario": name,
        "in_size": in_size,
        "median_ms": _median_ms(samples),
        "samples_ms": [round(value, 2) for value in samples],
    }


def run_benchmark() -> list[dict[str, Any]]:
    random.seed(SEED)
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "in_clause_benchmark.db"
        _seed_sqlite(db_path)

        helper = DataHelper(
            backend="sqlalchemy",
            connection_url=f"sqlite:///{db_path}",
            worker_connection_env_var="BENCH_DB_DSN",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
            table="events",
        )
        try:
            os.environ["BENCH_DB_DSN"] = f"sqlite:///{db_path}"
            id_pool = list(range(1, ROW_COUNT + 1))
            scenarios = _build_scenarios(helper)

            results: list[dict[str, Any]] = []
            for in_size in IN_SIZES:
                ids = random.sample(id_pool, in_size)
                for name, scenario in scenarios.items():
                    results.append(_benchmark_scenario(name, scenario, ids, in_size))
            return results
        finally:
            helper.close()


def main() -> None:
    results = run_benchmark()
    print("scenario,in_size,median_ms,samples_ms")
    for row in results:
        print(f"{row['scenario']},{row['in_size']},{row['median_ms']},{row['samples_ms']}")


if __name__ == "__main__":
    main()
