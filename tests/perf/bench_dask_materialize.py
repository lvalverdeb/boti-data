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
from typing import Any

from boti_data.helper import DataHelper

ITERATIONS = 5
COLUMNS = ("id_producto", "cliente_id", "product_type_id", "global_track_id")
TABLE = "asm_tracking_productos"


def _load_env(env_file: Path) -> None:
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_file)
        print(f"  loaded env from {env_file}")


def _build_config(**overrides: str) -> dict[str, Any]:
    base: dict[str, Any] = {
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


# Not a copy-pasted twin: the only difference is fn() vs await fn() — a
# genuine sync/async call difference for a benchmark helper measuring both
# code paths, not copy-paste.
# spaghetti-ignore[sync-async-duplication]
def _measure_sync(label: str, fn, iterations: int) -> list[float]:
    times: list[float] = []
    for i in range(iterations):
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"    [{label}] iteration {i + 1}/{iterations}: {elapsed:.3f}s")
    return times


async def _measure_async(label: str, fn, iterations: int) -> list[float]:
    times: list[float] = []
    for i in range(iterations):
        t0 = time.perf_counter()
        await fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"    [{label}] iteration {i + 1}/{iterations}: {elapsed:.3f}s")
    return times


def _report(tag: str, times: list[float]) -> None:
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"\n  {tag}:")
    print(
        f"    mean={mean:.3f}s  min={min(times):.3f}s  max={max(times):.3f}s  "
        f"stdev={stdev:.3f}s  range={max(times) - min(times):.3f}s"
    )


def _report_triple(
    label: str, times_aload: list[float], times_compute: list[float], times_total: list[float]
) -> None:
    print()
    _report(f"{label} aload", times_aload)
    _report(f"{label} compute", times_compute)
    _report(f"{label} total", times_total)


class _AsyncGW:
    def __init__(self, dsn: str) -> None:
        self.gw = DataHelper(
            **_build_config(connection_url=dsn, worker_connection_env_var="ASYNC_DB_DSN")
        )

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


async def _bench_pandas_async(db_url_async: str, iterations: int) -> list[float]:
    async with _AsyncGW(db_url_async) as h:

        async def _load() -> None:
            df = await h.gw.pandas.aload(global_track_id__in=[5], columns=COLUMNS)
            _ = len(df)

        return await _measure_async("pandas", _load, iterations)


async def _bench_polars_async(db_url_async: str, iterations: int) -> list[float]:
    async with _AsyncGW(db_url_async) as h:

        async def _load() -> None:
            df = await h.gw.polars.aload(global_track_id__in=[5], columns=COLUMNS)
            _ = len(df)

        return await _measure_async("polars", _load, iterations)


# Not a copy-pasted twin: measures the same dask.aload()+compute() path over
# an async DSN vs a sync DSN (genuinely different connection/engine setup),
# not copy-paste.
# spaghetti-ignore[sync-async-duplication]
async def _bench_dask_async(
    db_url_async: str, iterations: int
) -> tuple[list[float], list[float], list[float]]:
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
            print(
                f"    [dask async] iteration {i + 1}/{iterations}: "
                f"aload={aload:.3f}s  compute={compute:.3f}s  total={total:.3f}s  "
                f"rows={len(pdf)}"
            )
        return times_aload, times_compute, times_total


def _bench_dask_sync(
    db_url_sync: str, iterations: int
) -> tuple[list[float], list[float], list[float]]:
    with _SyncGW(db_url_sync) as h:
        times_aload: list[float] = []
        times_compute: list[float] = []
        times_total: list[float] = []
        for i in range(iterations):
            t0 = time.perf_counter()
            result = h.gw.dask.load(global_track_id__in=[5], columns=COLUMNS)
            t1 = time.perf_counter()
            pdf = result.compute()
            t2 = time.perf_counter()
            aload = t1 - t0
            compute = t2 - t1
            total = t2 - t0
            times_aload.append(aload)
            times_compute.append(compute)
            times_total.append(total)
            print(
                f"    [dask sync] iteration {i + 1}/{iterations}: "
                f"aload={aload:.3f}s  compute={compute:.3f}s  total={total:.3f}s  "
                f"rows={len(pdf)}"
            )
    return times_aload, times_compute, times_total


def _summary_row(label: str, times: list[float]) -> list[str]:
    stdev = f"{statistics.stdev(times):.3f}s" if len(times) > 1 else "N/A"
    return [
        label,
        f"{statistics.mean(times):.3f}s",
        f"{min(times):.3f}s",
        f"{max(times):.3f}s",
        stdev,
        f"{max(times) - min(times):.3f}s",
    ]


def _build_summary_rows(
    times_pd: list[float],
    times_pl: list[float],
    dask_async_t: tuple[list[float], list[float], list[float]],
    times_aload_sync: list[float],
    times_compute_sync: list[float],
    times_total_sync: list[float],
) -> list[list[str]]:
    return [
        _summary_row("pandas aload", times_pd),
        _summary_row("polars aload", times_pl),
        _summary_row("dask async aload", dask_async_t[0]),
        _summary_row("dask async compute", dask_async_t[1]),
        _summary_row("dask async total", dask_async_t[2]),
        _summary_row("dask sync aload", times_aload_sync),
        _summary_row("dask sync compute", times_compute_sync),
        _summary_row("dask sync total", times_total_sync),
    ]


def _print_summary_table(rows: list[list[str]]) -> None:
    headers = ["method", "mean", "min", "max", "stdev", "range"]
    col_widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "  ".join("-" * w for w in col_widths)
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, col_widths)))


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

    print("\n--- BENCH: pandas (async DSN) ---")
    times_pd = asyncio.run(_bench_pandas_async(db_url_async, iterations))
    _report("pandas aload", times_pd)

    print("\n--- BENCH: polars (async DSN) ---")
    times_pl = asyncio.run(_bench_polars_async(db_url_async, iterations))
    _report("polars aload", times_pl)

    print("\n--- BENCH: dask (async DSN) ---")
    dask_async_t = asyncio.run(_bench_dask_async(db_url_async, iterations))
    _report_triple("dask async", *dask_async_t)

    print("\n--- BENCH: dask (sync DSN) ---")
    times_aload_sync, times_compute_sync, times_total_sync = _bench_dask_sync(
        db_url_sync, iterations
    )
    _report_triple("dask sync", times_aload_sync, times_compute_sync, times_total_sync)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    rows = _build_summary_rows(
        times_pd, times_pl, dask_async_t, times_aload_sync, times_compute_sync, times_total_sync
    )
    _print_summary_table(rows)

    print("\nDone.")


if __name__ == "__main__":
    import sys

    kwargs = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--iterations="):
            kwargs["iterations"] = int(arg.split("=", 1)[1])
    main(**kwargs)
