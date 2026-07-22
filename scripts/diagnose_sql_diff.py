"""Compare the SQL statements produced by the dask and pandas paths.

Usage:
    python scripts/diagnose_sql_diff.py

Requires env vars SYNC_DB_DSN or ASYNC_DB_DSN with a working MySQL connection.
"""

import os
import sys
from pathlib import Path
from types import MappingProxyType

# Ensure we can resolve the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env.local
dotenv_path = Path(__file__).resolve().parent.parent / ".env.local"
if dotenv_path.exists():
    for line in dotenv_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Pick a DSN — prefer SYNC, fall back to ASYNC (stripping async driver prefix)
dsn = os.getenv("SYNC_DB_DSN")
if not dsn:
    async_dsn = os.getenv("ASYNC_DB_DSN", "")
    if async_dsn.startswith("mysql+aiomysql://"):
        dsn = "mysql+pymysql://" + async_dsn[len("mysql+aiomysql://") :]
if not dsn:
    print("ERROR: Set SYNC_DB_DSN or ASYNC_DB_DSN")
    sys.exit(1)

print(f"DSN: {dsn.rsplit('@', 1)[0].rsplit(':', 1)[0]}:***@...")

from sqlalchemy import text
from sqlalchemy.dialects import mysql as mysql_dialect

from boti_data.db.partitioned_planner import SqlPartitionPlanner
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.db.sql_resource import SqlDatabaseResource
from boti_data.gateway.loaders import (
    _prepare_sql_statement,
    build_sql_partitioned_request,
    reflect_and_select,
)
from boti_data.gateway.normalization import build_partitioned_load_options
from boti_data.gateway.requests import SqlLoadRequest

# -- Configuration (mirrors what the benchmark notebook sets up) --
TABLE = "asm_tracking_productos"
DB_COLUMNS = ("id_track_global", "id_producto", "cliente_id", "id_tipo_producto")
FILTERS = MappingProxyType({"id_track_global__in": [1, 2, 3, 4], "id_tipo_producto": 1})
CHUNK_SIZE = 50000

config = SqlDatabaseConfig(
    connection_url=dsn,
    poolclass="sqlalchemy.pool.NullPool",
    query_only=True,
)

with SqlDatabaseResource(config) as res:
    # 1. Reflect + build base statement (shared by both paths)
    model, base_stmt = reflect_and_select(res, TABLE, DB_COLUMNS)
    engine = res.engine

    print(f"\nModel: {model.__name__ if hasattr(model, '__name__') else type(model).__name__}")
    print(f"Base statement selected_columns: {[c.name for c in base_stmt.selected_columns]}")

    # ===== DASK PATH =====
    planner = SqlPartitionPlanner(res)
    opts = build_partitioned_load_options(
        statement=base_stmt,
        model=model,
        filters=FILTERS,
        control={},
        default_chunk_size=CHUNK_SIZE,
    )
    req = build_sql_partitioned_request(opts.model_dump(exclude_none=True))
    dask_base = planner.prepare_statement(req)
    dask_limited = dask_base  # No LIMIT — avoids bad MySQL query plan

    # ===== PANDAS (EAGER) PATH =====
    pandas_req = SqlLoadRequest.model_validate(
        dict(
            statement=base_stmt,
            model=model,
            filters=FILTERS,
            as_pandas=True,
        )
    )
    pandas_stmt, _ = _prepare_sql_statement(pandas_req, logger=None, debug=False)

    # ===== COMPILE TO MYSQL DIALECT =====
    dialect = mysql_dialect.dialect()

    print("\n" + "=" * 72)
    print("COMPILED SQL — DASK PATH (no LIMIT — after fix)")
    print("=" * 72)
    dask_sql = str(dask_limited.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
    print(dask_sql)

    print("\n" + "=" * 72)
    print("COMPILED SQL — PANDAS PATH (no LIMIT)")
    print("=" * 72)
    pandas_sql = str(pandas_stmt.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
    print(pandas_sql)

    # ===== RUN EXPLAIN ON BOTH =====
    print("\n" + "=" * 72)
    print("EXPLAIN — DASK PATH")
    print("=" * 72)
    with engine.connect() as conn:
        result = conn.execute(text("EXPLAIN " + dask_sql))
        for row in result:
            print(row)

    print("\n" + "=" * 72)
    print("EXPLAIN — PANDAS PATH")
    print("=" * 72)
    with engine.connect() as conn:
        result = conn.execute(text("EXPLAIN " + pandas_sql))
        for row in result:
            print(row)

    # ===== TIMING COMPARISON =====
    import time

    print("\n" + "=" * 72)
    print("TIMING — 3 runs each (cold start)")
    print("=" * 72)

    def time_query(sql: str, label: str, runs: int = 3) -> list[float]:
        times = []
        for i in range(runs):
            with engine.connect() as conn:
                t0 = time.perf_counter()
                result = conn.exec_driver_sql(sql)
                rows = result.fetchall()
                t1 = time.perf_counter()
                times.append(t1 - t0)
                print(f"  {label} run {i + 1}: {t1 - t0:.4f}s ({len(rows)} rows)")
        return times

    dask_times = time_query(dask_sql, "dask")
    pandas_times = time_query(pandas_sql, "pandas")

    dask_avg = sum(dask_times) / len(dask_times)
    pandas_avg = sum(pandas_times) / len(pandas_times)
    ratio = dask_avg / pandas_avg if pandas_avg > 0 else float("inf")

    print("\n--- Summary ---")
    print(f"Dask avg:   {dask_avg:.4f}s")
    print(f"Pandas avg: {pandas_avg:.4f}s")
    print(f"Ratio:      {ratio:.2f}x")
    if ratio > 1.1:
        print(">>> The LIMIT clause is causing a meaningful slowdown!")
    elif ratio < 0.9:
        print(">>> The lack of LIMIT is causing a meaningful slowdown!")
    else:
        print(">>> The SQL difference is NOT the root cause. The gap is elsewhere.")
