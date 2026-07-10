"""
Standalone benchmark comparing pandas, polars, and dask (async + sync DSN) SQL
materialization times. Runs outside Jupyter.

Usage:
    uv run python tests/perf/bench_dask_materialize.py
    uv run python tests/perf/bench_dask_materialize.py --iterations 10
"""
from __future__ import annotations

import asyncio
import os
import statistics
import time
from pathlib import Path

from boti_data.helper import DataHelper

ITERATIONS = 5
COLUMNS = ["id_producto", "cliente_id", "product_type_id", "global_track_id"]
TABLE = "asm_tracking_productos"


def _load_env(env_file: Path) -> None:
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file)
        print(f"  loaded env from {env_file}")


def _build_config(**overrides: str) -> dict:
    base: dict = {
        "backend": "sqlalchemy",
        "poolclass": "sqlalchemy.pool.NullPool",
        "query_only": True,
        "table": TABLE,
        "field_map": {
            "id_track_global": "global_track_id",
            "id_tipo_producto": "product_type_id",
        },
        "sticky_filters": {
            "product_type_id": 1,
        },
    }
    base.update(overrides)
    return base


def _measure_sync(label: str, fn, iterations: int) -> list[float]:
    times: list[float] = []
    for i in range(iterations):
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"    [{label}] iteration {i+1}/{iterations}: {elapsed:.3f}s")
    return times


async def _measure_async(label: str, fn, iterations: int) -> list[float]:
    times: list[float] = []
    for i in range(iterations):
        t0 = time.perf_counter()
        await fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"    [{label}] iteration {i+1}/{iterations}: {elapsed:.3f}s")
    return times


def _report(tag: str, times: list[float]) -> None:
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"\n  {tag}:")
    print(f"    mean={mean:.3f}s  min={min(times):.3f}s  max={max(times):.3f}s  "
          f"stdev={stdev:.3f}s  range={max(times)-min(times):.3f}s")


def _report_triple(label: str, times_aload: list[float],
                   times_compute: list[float], times_total: list[float]) -> None:
    print()
    _report(f"{label} aload", times_aload)
    _report(f"{label} compute", times_compute)
    _report(f"{label} total", times_total)


def main(iterations: int = ITERATIONS) -> None:
    env_local = Path(__file__).resolve().parents[2] / ".env.local"
    if not env_local.exists():
        env_local = Path("/Users/lvalverdeb/TeamDev/repo-split/boti-data/.env.local")
    _load_env(env_local)

    db_url_async = os.getenv("ASYNC_DB_DSN")
    db_url_sync = os.getenv("SYNC_DB_DSN")
    if not db_url_async or not db_url_sync:
        raise RuntimeError("ASYNC_DB_DSN and SYNC_DB_DSN must be set")

    print("=" * 72)
    print("DASK MATERIALIZATION BENCHMARK")
    print(f"  iterations: {iterations}")
    print(f"  columns: {COLUMNS}")
    print("  filter: global_track_id__in=[5] (1.7M rows)")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # Helper classes for DataHelper lifecycle
    # ------------------------------------------------------------------ #
    class _AsyncGW:
        def __init__(self, dsn: str) -> None:
            self.gw = DataHelper(**_build_config(connection_url=dsn, worker_connection_env_var="ASYNC_DB_DSN"))

        async def __aenter__(self) -> _AsyncGW:
            return self

        async def __aexit__(self, *_: object) -> None:
            await asyncio.to_thread(self.gw.close)

    class _SyncGW:
        def __init__(self, dsn: str) -> None:
            self.gw = DataHelper(**_build_config(connection_url=dsn))

        def __enter__(self) -> _SyncGW:
            return self

        def __exit__(self, *_: object) -> None:
            self.gw.close()

    # ================================================================== #
    # 1) pandas aload (async DSN, single-fetch)
    # ================================================================== #
    async def bench_pandas_async() -> list[float]:
        async with _AsyncGW(db_url_async) as h:
            async def _load() -> None:
                df = await h.gw.pandas.aload(global_track_id__in=[5], columns=COLUMNS)
                _ = len(df)
            return await _measure_async("pandas", _load, iterations)

    print("\n--- BENCH: pandas (async DSN) ---")
    times_pd: list[float] = asyncio.run(bench_pandas_async())
    _report("pandas aload", times_pd)

    # ================================================================== #
    # 2) polars aload (async DSN, single-fetch)
    # ================================================================== #
    async def bench_polars_async() -> list[float]:
        async with _AsyncGW(db_url_async) as h:
            async def _load() -> None:
                df = await h.gw.polars.aload(global_track_id__in=[5], columns=COLUMNS)
                _ = len(df)
            return await _measure_async("polars", _load, iterations)

    print("\n--- BENCH: polars (async DSN) ---")
    times_pl: list[float] = asyncio.run(bench_polars_async())
    _report("polars aload", times_pl)

    # ================================================================== #
    # 3) dask aload (async DSN) + compute
    # ================================================================== #
    async def bench_dask_async() -> tuple[list[float], list[float], list[float]]:
        async with _AsyncGW(db_url_async) as h:
            times_aload: list[float] = []
            times_compute: list[float] = []
            times_total: list[float] = []
            for i in range(iterations):
                t0 = time.perf_counter()
                result = await h.gw.dask.aload(global_track_id__in=[5], columns=COLUMNS)
                t1 = time.perf_counter()
                pdf = await asyncio.to_thread(result.compute)
                t2 = time.perf_counter()
                aload = t1 - t0
                compute = t2 - t1
                total = t2 - t0
                times_aload.append(aload)
                times_compute.append(compute)
                times_total.append(total)
                print(f"    [dask async] iteration {i+1}/{iterations}: "
                      f"aload={aload:.3f}s  compute={compute:.3f}s  total={total:.3f}s  "
                      f"rows={len(pdf)}")
            return times_aload, times_compute, times_total

    print("\n--- BENCH: dask (async DSN) ---")
    dask_async_t = asyncio.run(bench_dask_async())
    _report_triple("dask async", *dask_async_t)

    # ================================================================== #
    # 4) dask aload (sync DSN) + compute
    # ================================================================== #
    print("\n--- BENCH: dask (sync DSN) ---")
    with _SyncGW(db_url_sync) as h:
        times_aload_sync: list[float] = []
        times_compute_sync: list[float] = []
        times_total_sync: list[float] = []
        for i in range(iterations):
            t0 = time.perf_counter()
            result = h.gw.dask.load(global_track_id__in=[5], columns=COLUMNS)
            t1 = time.perf_counter()
            pdf = result.compute()
            t2 = time.perf_counter()
            aload = t1 - t0
            compute = t2 - t1
            total = t2 - t0
            times_aload_sync.append(aload)
            times_compute_sync.append(compute)
            times_total_sync.append(total)
            print(f"    [dask sync] iteration {i+1}/{iterations}: "
                  f"aload={aload:.3f}s  compute={compute:.3f}s  total={total:.3f}s  "
                  f"rows={len(pdf)}")
    _report_triple("dask sync", times_aload_sync, times_compute_sync, times_total_sync)

    # ================================================================== #
    # Summary table
    # ================================================================== #
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    headers = ["method", "mean", "min", "max", "stdev", "range"]
    rows: list[list[str]] = [
        ["pandas aload",
         f"{statistics.mean(times_pd):.3f}s",
         f"{min(times_pd):.3f}s",
         f"{max(times_pd):.3f}s",
         f"{statistics.stdev(times_pd):.3f}s" if len(times_pd) > 1 else "N/A",
         f"{max(times_pd)-min(times_pd):.3f}s"],
        ["polars aload",
         f"{statistics.mean(times_pl):.3f}s",
         f"{min(times_pl):.3f}s",
         f"{max(times_pl):.3f}s",
         f"{statistics.stdev(times_pl):.3f}s" if len(times_pl) > 1 else "N/A",
         f"{max(times_pl)-min(times_pl):.3f}s"],
        ["dask async aload",
         f"{statistics.mean(dask_async_t[0]):.3f}s",
         f"{min(dask_async_t[0]):.3f}s",
         f"{max(dask_async_t[0]):.3f}s",
         f"{statistics.stdev(dask_async_t[0]):.3f}s",
         f"{max(dask_async_t[0])-min(dask_async_t[0]):.3f}s"],
        ["dask async compute",
         f"{statistics.mean(dask_async_t[1]):.3f}s",
         f"{min(dask_async_t[1]):.3f}s",
         f"{max(dask_async_t[1]):.3f}s",
         f"{statistics.stdev(dask_async_t[1]):.3f}s",
         f"{max(dask_async_t[1])-min(dask_async_t[1]):.3f}s"],
        ["dask async total",
         f"{statistics.mean(dask_async_t[2]):.3f}s",
         f"{min(dask_async_t[2]):.3f}s",
         f"{max(dask_async_t[2]):.3f}s",
         f"{statistics.stdev(dask_async_t[2]):.3f}s",
         f"{max(dask_async_t[2])-min(dask_async_t[2]):.3f}s"],
        ["dask sync aload",
         f"{statistics.mean(times_aload_sync):.3f}s",
         f"{min(times_aload_sync):.3f}s",
         f"{max(times_aload_sync):.3f}s",
         f"{statistics.stdev(times_aload_sync):.3f}s",
         f"{max(times_aload_sync)-min(times_aload_sync):.3f}s"],
        ["dask sync compute",
         f"{statistics.mean(times_compute_sync):.3f}s",
         f"{min(times_compute_sync):.3f}s",
         f"{max(times_compute_sync):.3f}s",
         f"{statistics.stdev(times_compute_sync):.3f}s",
         f"{max(times_compute_sync)-min(times_compute_sync):.3f}s"],
        ["dask sync total",
         f"{statistics.mean(times_total_sync):.3f}s",
         f"{min(times_total_sync):.3f}s",
         f"{max(times_total_sync):.3f}s",
         f"{statistics.stdev(times_total_sync):.3f}s",
         f"{max(times_total_sync)-min(times_total_sync):.3f}s"],
    ]

    col_widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "  ".join("-" * w for w in col_widths)
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, col_widths)))

    print("\nDone.")


if __name__ == "__main__":
    import sys
    kwargs = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--iterations="):
            kwargs["iterations"] = int(arg.split("=", 1)[1])
    main(**kwargs)
