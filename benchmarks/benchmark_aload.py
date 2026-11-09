import asyncio
import os
import statistics
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import Column, Integer, select
from sqlalchemy.orm import declarative_base

from boti_data.helper import DataHelper

env_file = Path(__file__).resolve().parents[1] / ".env.local"
load_dotenv(dotenv_path=env_file)

db_url_async = os.getenv("ASYNC_DB_DSN")

Row = dict[str, Any]

BASE_CONFIG = MappingProxyType(
    {
        "backend": "sqlalchemy",
        "connection_url": db_url_async,
        "worker_connection_env_var": "ASYNC_DB_DSN",
        "poolclass": "sqlalchemy.pool.NullPool",
        "query_only": True,
        "table": "asm_tracking_productos",
        "field_map": {"id_track_global": "global_track_id", "id_tipo_producto": "product_type_id"},
        "sticky_filters": {"product_type_id": 1},
    }
)

COLUMNS = ("id_producto", "cliente_id", "product_type_id", "global_track_id")

CONFIGS: list[tuple[str, str, int | None]] = [
    ("pandas", "pandas", None),
    ("polars", "polars", None),
    ("dask_default", "dask", None),
    ("dask_single", "dask", 2_000_000),
    ("dask_7p", "dask", 250_000),
    ("dask_18p", "dask", 100_000),
    ("dask_50k", "dask", 50_000),
]

FILTER_SETS: list[Row] = [
    # dict(global_track_id__in=[1, 2, 3, 4]),
    dict(global_track_id__in=[5]),
]
REPEATS = 3


async def benchmark(label: str, backend: str, chunk_size: int | None, filters: Row) -> Row:
    gateway = DataHelper(**BASE_CONFIG)
    try:
        aload_times: list[float] = []
        compute_times: list[float] = []
        total_times: list[float] = []
        rows: int = 0
        npartitions: int = 1

        for i in range(REPEATS):
            kwargs = {**filters, "columns": COLUMNS}
            if chunk_size is not None:
                kwargs["chunk_size"] = chunk_size

            t0 = time.perf_counter()
            if backend == "dask":
                result = await gateway.dask.aload(**kwargs)
                t1 = time.perf_counter()
                npartitions = result.npartitions
                result = result.compute()
                t2 = time.perf_counter()
            else:
                result = await getattr(gateway, backend).aload(**kwargs)
                t1 = t2 = time.perf_counter()
                npartitions = 1

            aload_times.append(t1 - t0)
            compute_times.append(t2 - t1)
            total_times.append(t2 - t0)
            rows = len(result)
            print(
                f"  [{label}] run {i + 1}: "
                f"aload={t1 - t0:.3f}s  "
                f"{'compute=' + f'{t2 - t1:.3f}s  ' if backend == 'dask' else 'eager    '}"
                f"total={t2 - t0:.3f}s  npart={npartitions}  rows={len(result)}"
            )

        return {
            "label": label,
            "backend": backend,
            "aload_mean": statistics.mean(aload_times),
            "compute_mean": statistics.mean(compute_times),
            "total_mean": statistics.mean(total_times),
            "rows": rows,
            "npartitions": npartitions,
        }
    finally:
        gateway.close()


def fmt(val: float) -> str:
    return f"{val:.3f}s"


# Structured-path model, matches asm_tracking_productos.
_Base = declarative_base()


class _TrackingProduct(_Base):
    __tablename__ = "asm_tracking_productos"
    id_producto = Column(Integer, primary_key=True, autoincrement=True)
    cliente_id = Column(Integer)
    id_track_global = Column(Integer)
    id_tipo_producto = Column(Integer)


# use_arrow comparison via the structured path.
_COLUMNS_MAP = {
    "id_producto": _TrackingProduct.id_producto,
    "cliente_id": _TrackingProduct.cliente_id,
    "id_track_global": _TrackingProduct.id_track_global,
    "id_tipo_producto": _TrackingProduct.id_tipo_producto,
}

_COL_KEYS = list(_COLUMNS_MAP.keys())


def _build_structured_stmt(filters: Row) -> Any:
    """Build a select statement matching the configured-path filters."""
    clauses: list = []
    for k, v in filters.items():
        if k == "global_track_id__in":
            clauses.append(_TrackingProduct.id_track_global.in_(v))
        else:
            raise ValueError(f"Unknown filter key: {k}")
    # sticky_filters applied manually in structured mode
    clauses.append(_TrackingProduct.id_tipo_producto == 1)
    col_exprs = [_COLUMNS_MAP[c] for c in _COL_KEYS]
    return select(*col_exprs).where(*clauses)


async def _measure_use_arrow_variant(base: Row, stmt: Any, use_arrow: bool) -> tuple[Row, int]:
    store: dict[str, list[float]] = {"aload": [], "compute": [], "total": []}
    rows = 0
    for _ in range(REPEATS):
        gw = DataHelper(**base)
        try:
            kwargs: Row = dict(statement=stmt, model=_TrackingProduct, columns=_COL_KEYS)
            if not use_arrow:
                kwargs["use_arrow"] = False

            t0 = time.perf_counter()
            result = await gw.dask.aload(**kwargs)
            t1 = time.perf_counter()
            df = result.compute()
            t2 = time.perf_counter()
        finally:
            gw.close()

        store["aload"].append(t1 - t0)
        store["compute"].append(t2 - t1)
        store["total"].append(t2 - t0)
        rows = len(df)
    return store, rows


def _print_use_arrow_comparison(
    filter_desc: Any,
    rows: int,
    arrow_results: dict[str, list[float]],
    noarrow_results: dict[str, list[float]],
) -> None:
    def mean(vals: list[float]) -> float:
        return statistics.mean(vals)

    arrow_a, arrow_c, arrow_t = (
        mean(arrow_results["aload"]),
        mean(arrow_results["compute"]),
        mean(arrow_results["total"]),
    )
    noarrow_a, noarrow_c, noarrow_t = (
        mean(noarrow_results["aload"]),
        mean(noarrow_results["compute"]),
        mean(noarrow_results["total"]),
    )

    print(f"  FILTER {filter_desc} — structured path ({rows:,} rows)")
    print(f"  {'─' * 66}")
    print(f"  {'use_arrow':<18} {'aload':>10} {'compute':>10} {'total':>10}")
    print(f"  {'─' * 66}")
    print(f"  {'True  (default)':<18} {fmt(arrow_a):>10} {fmt(arrow_c):>10} {fmt(arrow_t):>10}")
    print(
        f"  {'False (explicit)':<18} {fmt(noarrow_a):>10} {fmt(noarrow_c):>10} {fmt(noarrow_t):>10}"
    )
    print(f"  {'─' * 66}")
    diff = arrow_c - noarrow_c
    max_c = max(arrow_c, noarrow_c)
    pct = abs(diff) / max_c * 100 if max_c > 0.001 else 0
    if max_c < 0.5:
        verdict = f"IDENTICAL — fast path catches both (compute ≈ {max_c:.3f}s)"
    elif abs(diff) < 0.5:
        verdict = f"SIMILAR (gap {abs(diff):.3f}s, {pct:.0f}%)"
    else:
        verdict = (
            f"DIFFERENT by {abs(diff):.3f}s ({pct:.0f}%) — Arrow-native "
            f"fetch vs pd.read_sql conversion, not the gate"
        )
    print(f"  Compute-gap: {diff:+.3f}s  →  {verdict}")
    print()


async def compare_use_arrow() -> None:
    """Compare ``use_arrow=True`` vs ``False`` on the structured path
    (``DataGateway.aload(statement=..., model=...)``).

    ``use_arrow`` selects only the *fetch representation* — arrow-native fetch
    vs a ``pd.read_sql`` conversion — it does not decide which execution path
    runs. Both settings take the same adaptive single-fetch path and, for a
    result under the memory-budget ceiling, both return a single partition in one
    scan. Any gap here is the arrow-vs-pandas fetch cost, not a structural
    difference in planning.
    """
    # Structured-mode gateway: no *table* → _configured=False
    base = {
        k: v for k, v in BASE_CONFIG.items() if k not in ("table", "field_map", "sticky_filters")
    }

    for fi, filters in enumerate(FILTER_SETS):
        filter_desc = list(filters.values())[0]
        stmt = _build_structured_stmt(filters)

        arrow_results, rows = await _measure_use_arrow_variant(base, stmt, True)
        noarrow_results, rows = await _measure_use_arrow_variant(base, stmt, False)

        _print_use_arrow_comparison(filter_desc, rows, arrow_results, noarrow_results)

    print("  Conclusion: use_arrow only changes the fetch representation (arrow-native")
    print("  vs pd.read_sql), not which path runs — both take the adaptive single-fetch")
    print("  path and single-scan under the ceiling.")
    print()


def _print_parity_table(eager: list[Row], dask_rows: list[Row], baseline: float | None) -> None:
    print(f"  {'Config':<16}{'aload':>9}{'compute':>9}{'total':>9}{'npart':>7}{'vs floor':>11}")
    print(f"  {'─' * 68}")
    for r in eager + dask_rows:
        ratio = f"{r['total_mean'] / baseline:.2f}x" if baseline else "—"
        print(
            f"  {r['label']:<16}{r['aload_mean']:>8.2f}s{r['compute_mean']:>8.2f}s"
            f"{r['total_mean']:>8.2f}s{r.get('npartitions', 1):>7}{ratio:>11}"
        )
    print(f"  {'─' * 68}")


def _report_single_scan_parity(single: list[Row], baseline: float, tol: float) -> None:
    if not single:
        return
    # Core guarantee: single-scan configs must reach parity with the eager floor.
    single_bad = [r for r in single if r["total_mean"] > baseline * tol]
    if not single_bad:
        print(
            f"  ✓ single-scan configs (npart=1) at parity with eager "
            f"(≤{tol:.2f}x floor): " + ", ".join(r["label"] for r in single)
        )
        return
    worst = max(single_bad, key=lambda r: r["total_mean"])
    print(
        f"  ✗ single-scan config {worst['label']} is "
        f"{worst['total_mean'] / baseline:.2f}x the eager floor — the fast "
        f"path is NOT delivering parity (regression?)"
    )


def _report_partitioned_configs(parted: list[Row], baseline: float) -> None:
    if not parted:
        return
    print(
        "  • partitioned configs (npart>1) slower by design: "
        + ", ".join(f"{r['label']} {r['total_mean'] / baseline:.2f}x" for r in parted)
    )


def _report_filter_set_parity(filters: Row, results: list[Row], *, tol: float) -> None:
    rows = results[0]["rows"]
    eager = [r for r in results if r.get("backend") != "dask"]
    dask_rows = [r for r in results if r.get("backend") == "dask"]
    if not dask_rows:
        return
    baseline = min((r["total_mean"] for r in eager), default=None)
    filter_desc = list(filters.values())[0]

    header = f"  FILTER={filter_desc}  rows={rows:,}"
    if baseline is not None:
        header += f"  eager_floor={baseline:.2f}s"
    print(header)
    print(f"  {'─' * 68}")
    _print_parity_table(eager, dask_rows, baseline)

    if baseline is None:
        print("  (no eager baseline in this run — add pandas/polars configs)\n")
        return

    single = [r for r in dask_rows if r.get("npartitions", 1) == 1]
    parted = [r for r in dask_rows if r.get("npartitions", 1) > 1]
    _report_single_scan_parity(single, baseline, tol)
    _report_partitioned_configs(parted, baseline)
    print()


def analyze_parity(all_results: list[list[Row]]) -> None:
    """Report whether dask reaches parity with the eager (pandas/polars) backends.

    ``aload`` picks its strategy adaptively. When a result fits under the
    single-fetch ceiling it is streamed in one scan and returned as a single
    partition (``npart=1``); partitioning a result that fits in one process is
    pure overhead — a COUNT round-trip plus per-partition fetch coordination — so
    there dask should *match* the eager backends. When a result overflows the
    ceiling, dask partitions (``npart>1``) and is expected to be slower.

    The ceiling is a *memory* budget (~512 MB) converted to a row count via the
    estimated row width, not a fixed row count and not ``chunk_size``. So for a
    narrow schema most/all configs single-scan and reach parity regardless of the
    ``chunk_size`` each config requests.

    This gates on **total** time against the best eager baseline — deliberately
    not on ``compute`` alone. Fast materialize is necessary but not sufficient:
    the original LIMIT-probe regression lived entirely in ``aload`` while
    ``compute`` stayed ~0s, so a compute-only check reported "OK" on a ~3x
    slowdown. Comparing totals against the eager floor is what catches that.
    """
    # Within this multiple of the eager floor counts as parity. Kept generous
    # because these are identical single-scan operations against a live DB, where
    # run-to-run noise alone spans ~25%; the check exists to catch *gross*
    # regressions (the original LIMIT-probe bug was ~3x), not fine timing jitter.
    PARITY_TOL = 1.5

    print(f"\n{'=' * 72}")
    print("  PARITY ANALYSIS  (dask total vs eager pandas/polars floor)")
    print(f"{'=' * 72}\n")

    for fi, filters in enumerate(FILTER_SETS):
        _report_filter_set_parity(filters, all_results[fi], tol=PARITY_TOL)

    print("  Note: because the ceiling is a byte budget, a narrow schema single-scans")
    print("  most/all configs at parity regardless of chunk_size. To exercise the")
    print("  partitioned path, lower single_fetch_threshold or use a larger result.")
    print()


async def main() -> None:
    all_results: list[list[Row]] = []
    for filters in FILTER_SETS:
        filter_desc = list(filters.values())[0]
        print(f"\n{'=' * 72}")
        print(f"  FILTER: {filter_desc}")
        print(f"{'=' * 72}\n")

        h = ["Label", "aload", "materialize", "total", "rows"]
        cw = [max(len(h[i]), 20 if i == 0 else 14) for i in range(len(h))]
        print("  ".join(h[i].ljust(cw[i]) for i in range(len(h))))
        print("  ".join("-" * cw[i] for i in range(len(h))))

        results: list[Row] = []
        for label, backend, chunk_size in CONFIGS:
            r = await benchmark(label, backend, chunk_size, filters)
            results.append(r)
            vals = [
                r["label"],
                fmt(r["aload_mean"]),
                fmt(r["compute_mean"]),
                fmt(r["total_mean"]),
                str(r["rows"]),
            ]
            print("  ".join(vals[i].ljust(cw[i]) for i in range(len(vals))))
            print()

        # Relative breakdown
        print("  Aload vs Materialize breakdown:")
        print(f"    {'Label':<20} {'aload%':<10} {'mat%':<10} {'mat_overhead':<14}")
        for r in results:
            aload_pct = r["aload_mean"] / r["total_mean"] * 100
            mat_pct = r["compute_mean"] / r["total_mean"] * 100
            overhead = r["compute_mean"] - r["aload_mean"]
            sign = "+" if overhead >= 0 else ""
            print(
                f"    {r['label']:<20} {aload_pct:<9.1f}% {mat_pct:<9.1f}% {sign}{fmt(abs(overhead))}"
            )
        print()

        all_results.append(results)

    analyze_parity(all_results)
    await compare_use_arrow()


if __name__ == "__main__":
    asyncio.run(main())
