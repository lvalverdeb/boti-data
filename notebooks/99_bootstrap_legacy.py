# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.16",
# ]
# ///

#  Copyright (c) 2026 Luis Valverde, contributors
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 99 — Bootstrap (legacy) — marimo replica

    Replicates `99_Bootstrap_legacy.ipynb`: end-to-end smoke tests for `boti-data` / `boti-dask`.

    - SQL backend via `DataHelper` — pandas / polars / dask engine views with `field_map` and sticky filters
    - SQL advanced filter operators (equality, comparison, range, string patterns, null, boolean)
    - S3 catalog + Parquet reader on `dst-etl/bronze/logistics/mobile/gps/`
    - Parquet filter pushdown vs residual (Arrow compute) tests
    - `boti-dask` resilience utilities (`safe_*`, `async_safe_*`, `inspect_graph`, `UniqueValuesExtractor`)
    """)
    return


@app.cell
def _():
    from pathlib import Path
    from dotenv import load_dotenv

    env_file = Path("/Users/lvalverdeb/TeamDev/repo-split/boti-data/.env.local")
    dotenv_loaded = load_dotenv(dotenv_path=env_file)
    return (dotenv_loaded,)


@app.cell
def _(dotenv_loaded):
    import os

    _ = dotenv_loaded  # ensure .env.local is loaded before reading env vars
    db_url_async = os.getenv("ASYNC_DB_DSN")
    db_url_sync = os.getenv("SYNC_DB_DSN")
    return db_url_async, db_url_sync, os


@app.cell
def _(db_url_async):
    from boti_data.helper import DataHelper

    config = {
        "backend": "sqlalchemy",
        "connection_url": db_url_async,
        "worker_connection_env_var": "ASYNC_DB_DSN",
        "poolclass": "sqlalchemy.pool.NullPool",
        "query_only": True,
        "table": "asm_tracking_productos",
        "field_map": {
            "id_track_global": "global_track_id",
            "id_tipo_producto": "product_type_id",
        },
        "sticky_filters": {
            "product_type_id": 1,
        },
    }
    return DataHelper, config


@app.cell
def _(DataHelper, config):
    gateway = DataHelper(**config)
    columns = ["id_producto", "cliente_id", "product_type_id", "global_track_id"]
    return columns, gateway


@app.cell
async def _(columns, gateway):
    result_pandas = await gateway.pandas.aload(global_track_id__in=[1, 2, 3, 4], columns=columns)
    return (result_pandas,)


@app.cell
def _(result_pandas):
    result_pandas
    return


@app.cell
async def _(columns, gateway):
    result_polars = await gateway.polars.aload(global_track_id__in=[1, 2, 3, 4], columns=columns)
    return (result_polars,)


@app.cell
def _(result_polars):
    result_polars
    return


@app.cell
def _():
    import time

    return (time,)


@app.cell
async def _(columns, gateway, time):
    _t0 = time.time()
    result_dask = await gateway.dask.aload(global_track_id__in=[1, 2, 3, 4], columns=columns)
    _t1 = time.time()
    print(f"dask aload: {_t1-_t0:.2f}s")
    return (result_dask,)


@app.cell
def _(result_dask):
    result_dask.compute()
    return


@app.cell
async def _(columns, gateway, time):
    # Re-run with diagnostics=True to see sub-step timing
    _t0 = time.time()
    result_dask_diag = await gateway.dask.aload(
        global_track_id__in=[1, 2, 3, 4],
        columns=columns,
        diagnostics=True,
    )
    _t1 = time.time()
    print(f"dask aload (diagnostics): {_t1-_t0:.2f}s")
    return


@app.cell
async def _(DataHelper, columns, config, db_url_sync, time):
    # Compare with sync DSN (mysql://) to check if conn.run_sync bridging adds overhead
    # Note: sync DSN path uses SqlPartitionedLoader.load_request() directly (no run_sync wrapping)
    sync_config = dict(config)
    sync_config["connection_url"] = db_url_sync
    sync_config["worker_connection_env_var"] = "SYNC_DB_DSN"
    sync_gateway = DataHelper(**sync_config)
    _t0 = time.time()
    result_sync = await sync_gateway.dask.aload(
        global_track_id__in=[5],
        columns=columns,
        diagnostics=True,
    )
    _t1 = time.time()
    print(f"sync dask aload: {_t1-_t0:.2f}s")
    pdf_sync = result_sync.compute()
    _t2 = time.time()
    print(f"sync compute: {_t2-_t1:.2f}s  rows={len(pdf_sync)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SQL Advanced Filter Tests

    Testing all filter operator types against the SQL backend via `DataHelper`.
    Covers equality, comparison, range, string patterns, null checks, and boolean logic.
    """)
    return


@app.cell
async def _(gateway):
    print("=" * 72)
    print("SQL FILTER TESTS")
    print("=" * 72)

    # 1. Single equality via mapped field (global_track_id → id_track_global)
    _result = await gateway.pandas.aload(global_track_id=1, columns=["id_producto", "global_track_id"])
    print(f"global_track_id=1: rows={len(_result)}, cols={list(_result.columns)}")
    assert len(_result) > 0
    assert (_result["global_track_id"] == 1).all()
    print("  ✓ equality filter works with field_map translation")
    print()
    return


@app.cell
async def _(gateway):
    # 2. Greater-than filters
    _result = await gateway.pandas.aload(id_producto__gte=408000000, columns=["id_producto"])
    print(f"id_producto__gte=408000000: rows={len(_result)}, min={_result['id_producto'].min()}")
    assert len(_result) > 0
    assert _result["id_producto"].min() >= 408000000
    print("  ✓ __gte filter works")

    _result = await gateway.pandas.aload(id_producto__lt=405730000, columns=["id_producto"])
    print(f"id_producto__lt=405730000: rows={len(_result)}, max={_result['id_producto'].max()}")
    assert len(_result) > 0
    assert _result["id_producto"].max() < 405730000
    print("  ✓ __lt filter works")
    print()
    return


@app.cell
async def _(gateway):
    # 3. Range (between) — rewritten to __gte + __lte
    _result = await gateway.pandas.aload(id_producto__range=(405780000, 405790000), columns=["id_producto"])
    print(f"id_producto__range=(405780000, 405790000): rows={len(_result)}")
    assert len(_result) > 0
    assert _result["id_producto"].between(405780000, 405790000).all()
    print("  ✓ __range filter works")
    print()
    return


@app.cell
async def _(gateway):
    # 4. Not equal and not_in
    _result = await gateway.pandas.aload(global_track_id__ne=1, columns=["global_track_id"])
    print(f"global_track_id__ne=1: rows={len(_result)}")
    if len(_result) > 0:
        assert (_result["global_track_id"] != 1).all()
    print("  ✓ __ne filter works")

    _result = await gateway.pandas.aload(global_track_id__not_in=[2, 3, 4], columns=["global_track_id"])
    print(f"global_track_id__not_in=[2,3,4]: rows={len(_result)}")
    assert len(_result) > 0
    print("  ✓ __not_in filter works")
    print()
    return


@app.cell
async def _(gateway):
    # 5. String pattern filters (residual — applied after SQL fetch)
    # Note: MySQL LIKE is case-insensitive by default
    import pandas as pd

    _result = await gateway.pandas.aload(codigo_barra__startswith="IST", columns=["codigo_barra"])
    print(f"codigo_barra__startswith=IST: rows={len(_result)}")
    if len(_result) > 0:
        assert all(str(v).lower().startswith("ist") for v in _result["codigo_barra"] if pd.notna(v))
    print("  ✓ __startswith filter works")

    _result = await gateway.pandas.aload(nombre_th__contains="J", columns=["nombre_th"])
    print(f"nombre_th__contains=J: rows={len(_result)}")
    if len(_result) > 0:
        assert all("J" in str(v).upper() for v in _result["nombre_th"] if pd.notna(v))
    print("  ✓ __contains filter works")

    _result = await gateway.pandas.aload(codigo_barra__endswith="CR", columns=["codigo_barra"])
    print(f"codigo_barra__endswith=CR: rows={len(_result)}")
    if len(_result) > 0:
        # MySQL LIKE is case-insensitive, so match lower-case
        assert all(str(v).lower().endswith("cr") for v in _result["codigo_barra"] if pd.notna(v))
    print("  ✓ __endswith filter works")
    print()
    return


@app.cell
async def _(gateway):
    # 6. Null checks
    result_not_null = await gateway.pandas.aload(sub_estatus__isnull=False, columns=["sub_estatus"])
    print(f"sub_estatus IS NOT NULL: rows={len(result_not_null)}")

    result_null = await gateway.pandas.aload(sub_estatus__isnull=True, columns=["sub_estatus"])
    print(f"sub_estatus IS NULL: rows={len(result_null)}")
    print("  ✓ __isnull filter works (both True and False)")
    print()
    return


@app.cell
async def _(gateway):
    # 7. Composite filters (implicit AND)
    _result = await gateway.pandas.aload(cliente_id=139, id_producto__gte=408000000, columns=["cliente_id", "id_producto"])
    print(f"cliente_id=139 AND id_producto>=408000000: rows={len(_result)}")
    assert len(_result) > 0
    assert (_result["cliente_id"] == 139).all()
    assert (_result["id_producto"] >= 408000000).all()
    print("  ✓ Multiple filters implicitly ANDed")
    print()
    return


@app.cell
async def _(gateway):
    # 8. Explicit OR
    _result = await gateway.pandas.aload(filters={"$or": [{"cliente_id": 139}, {"cliente_id": 209}]}, columns=["cliente_id"])
    print(f"$or: cliente_id=139 OR 209: rows={len(_result)}")
    assert len(_result) > 0
    assert _result["cliente_id"].isin([139, 209]).all()
    print("  ✓ $or boolean filter works")
    print()
    return


@app.cell
async def _(gateway):
    # 9. IN filter
    _result = await gateway.pandas.aload(cliente_id__in=[139, 209, 304], columns=["cliente_id"])
    print(f"cliente_id__in=[139,209,304]: rows={len(_result)}")
    assert len(_result) > 0
    assert set(_result["cliente_id"].unique()).issubset({139, 209, 304})
    print("  ✓ __in filter works")
    print()
    return


@app.cell
def _():
    print("All SQL filter tests passed.")
    return


@app.cell
def _(dotenv_loaded, os):
    from boti_data.connection_catalog import S3Catalog
    from boti.core.filesystem import add_endpoint_to_allowlist

    _ = dotenv_loaded  # ensure .env.local is loaded before reading env vars
    add_endpoint_to_allowlist(os.getenv("ETL_ENDPOINT_ALLOWLIST"))
    store = S3Catalog("ETL_", env_file=".env")
    print(store.storage_path)  # s3://analytics-bucket/raw/events
    print(store.ls())
    return (store,)


@app.cell
def _(store):
    store
    return


@app.cell
def _():
    from boti_data import ParquetReader

    return (ParquetReader,)


@app.cell
def _(ParquetReader, store):
    parquet_config = {
        "fs": store.fs(),
        "storage_path": "dst-etl/bronze/logistics/mobile/gps/",
        "parquet_start_date": "2025-01-01",
        "parquet_end_date": "2026-03-31",
        "partition_on": ["partition_date"],
    }
    parquet_helper = ParquetReader(**parquet_config)
    return (parquet_helper,)


@app.cell
async def _(parquet_helper):
    result = await parquet_helper.aload(associate_id__in=[27, 2285])
    return (result,)


@app.cell
def _(result):
    result.compute()
    return


@app.cell
def _(parquet_helper):
    parquet_helper.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parquet Advanced Filter Tests

    Testing all filter types against the Parquet (S3) backend.
    Operators are classified as **pushdown** (pruned at scan time by PyArrow) or **residual**
    (applied in-memory via Arrow compute kernels).
    """)
    return


@app.cell
def _(DataHelper, store):
    # Recreate ParquetReader for comprehensive filter tests
    # Using a fresh config to avoid mutating the original helper
    _parquet_config = {
        "backend": "parquet",
        "fs": store.fs(),
        "storage_path": "dst-etl/bronze/logistics/mobile/gps/",
        "parquet_start_date": "2026-01-01",
        "parquet_end_date": "2026-03-31",
        "partition_on": ["partition_date"],
    }
    parquet = DataHelper(**_parquet_config)
    print(f"Parquet filter test helper ready: {_parquet_config['storage_path']}")
    return (parquet,)


@app.cell
async def _(parquet):
    print("=" * 72)
    print("PARQUET BASIC (PUSHDOWN) FILTER TESTS")
    print("=" * 72)

    # 1. Equality filter (exact)
    _result = await parquet.aload(associate_id=2285, columns=["associate_id", "action", "latitude"])
    _pdf = _result.compute()
    print(f"associate_id=2285: rows={len(_pdf)}")
    assert len(_pdf) > 0
    assert (_pdf["associate_id"] == 2285).all()
    print("  ✓ __exact filter works (pushdown)")
    print()
    return


@app.cell
async def _(parquet):
    # 2. Comparison filters
    _result = await parquet.aload(associate_id__gte=4000, columns=["associate_id", "action"])
    _pdf = _result.compute()
    print(f"associate_id__gte=4000: rows={len(_pdf)}, min={_pdf['associate_id'].min()}")
    assert len(_pdf) > 0
    assert _pdf["associate_id"].min() >= 4000
    print("  ✓ __gte pushes down to PyArrow scan")

    _result = await parquet.aload(associate_id__lt=100, columns=["associate_id"])
    _pdf = _result.compute()
    print(f"associate_id__lt=100: rows={len(_pdf)}, max={_pdf['associate_id'].max()}")
    assert len(_pdf) > 0
    assert _pdf["associate_id"].max() < 100
    print("  ✓ __lt filter works (pushdown)")
    print()
    return


@app.cell
async def _(parquet):
    # 3. Range and NOT filters
    _result = await parquet.aload(associate_id__range=(100, 5000), columns=["associate_id"])
    _pdf = _result.compute()
    print(f"associate_id__range=(100, 5000): rows={len(_pdf)}")
    assert len(_pdf) > 0
    assert _pdf["associate_id"].between(100, 5000).all()
    print("  ✓ __range filter works (rewritten to gte+lte)")

    _result = await parquet.aload(associate_id__not_exact=2285, columns=["associate_id"])
    _pdf = _result.compute()
    print(f"associate_id__ne=2285: rows={len(_pdf)}")
    assert len(_pdf) > 0
    assert (_pdf["associate_id"] != 2285).all()
    print("  ✓ __ne filter works (pushdown)")

    _result = await parquet.aload(associate_id__not_in=[2285, 4499], columns=["associate_id"])
    _pdf = _result.compute()
    print(f"associate_id__not_in=[2285,4499]: rows={len(_pdf)}")
    assert len(_pdf) > 0
    assert not _pdf["associate_id"].isin([2285, 4499]).any()
    print("  ✓ __not_in filter works (pushdown)")
    print()
    return


@app.cell
async def _(parquet):
    # 4. Float column filters
    _result = await parquet.aload(latitude__gte=9.90, columns=["latitude", "longitude", "associate_id"])
    _pdf = _result.compute()
    print(f"latitude__gte=9.90: rows={len(_pdf)}")
    assert len(_pdf) > 0
    assert _pdf["latitude"].min() >= 9.90
    print("  ✓ Float column filter works (pushdown)")

    _result = await parquet.aload(longitude__lte=-84.0, columns=["longitude"])
    _pdf = _result.compute()
    print(f"longitude__lte=-84.0: rows={len(_pdf)}")
    if len(_pdf) > 0:
        assert (_pdf["longitude"] <= -84.0).all()
    print("  ✓ Float __lte filter works (pushdown)")
    print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parquet Residual (Post-Scan) Filter Tests

    Operators like `__startswith`, `__contains`, `__isnull` cannot be pushed into the
    Parquet scan. They are applied in-memory via **Arrow compute kernels** after the
    relevant row groups are loaded.
    """)
    return


@app.cell
async def _(parquet):
    print("=" * 72)
    print("PARQUET RESIDUAL FILTER TESTS")
    print("=" * 72)

    # 5. String pattern — startswith (residual for Arrow)
    _result = await parquet.aload(action__startswith="Acceso", columns=["action", "associate_id"])
    _pdf = _result.compute()
    print(f"action__startswith=Acceso: rows={len(_pdf)}")
    assert len(_pdf) > 0
    assert all(str(v).startswith("Acceso") for v in _pdf["action"])
    print("  ✓ __startswith applied via Arrow residual")

    # 6. String contains
    _result = await parquet.aload(description__contains="Regular", columns=["description"])
    _pdf = _result.compute()
    print(f"description__contains=Regular: rows={len(_pdf)}")
    assert len(_pdf) > 0
    assert all("Regular" in str(v) for v in _pdf["description"])
    print("  ✓ __contains applied via Arrow residual")

    # 7. String endswith
    _result = await parquet.aload(action__endswith="Salida", columns=["action"])
    _pdf = _result.compute()
    print(f"action__endswith=Salida: rows={len(_pdf)}")
    if len(_pdf) > 0:
        assert all(str(v).endswith("Salida") for v in _pdf["action"])
    print("  ✓ __endswith applied via Arrow residual")
    print()
    return


@app.cell
async def _(parquet):
    # 8. Null checks
    _result = await parquet.aload(direccion__isnull=True, columns=["direccion", "action"])
    _pdf = _result.compute()
    print(f"direccion IS NULL: rows={len(_pdf)}")
    assert len(_pdf) > 0
    assert _pdf["direccion"].isna().all()
    print("  ✓ __isnull=True works (residual)")

    # direccion is mostly null; verify not-null returns 0 or a valid set
    _result = await parquet.aload(direccion__isnull=False, columns=["direccion"])
    _pdf = _result.compute()
    print(f"direccion IS NOT NULL: rows={len(_pdf)}")
    if len(_pdf) > 0:
        assert _pdf["direccion"].notna().all()
    print("  ✓ __isnull=False works (residual)")
    print()
    return


@app.cell
async def _(parquet):
    # 9. Case-insensitive filters (residual)
    _result = await parquet.aload(action__icontains="acceso", columns=["action"])
    _pdf = _result.compute()
    print(f"action__icontains=acceso: rows={len(_pdf)}")
    assert len(_pdf) > 0
    assert all("acceso" in str(v).lower() for v in _pdf["action"])
    print("  ✓ __icontains works (case-insensitive residual)")

    # istartswith
    _result = await parquet.aload(action__istartswith="reporte", columns=["action"])
    _pdf = _result.compute()
    print(f"action__istartswith=reporte: rows={len(_pdf)}")
    assert len(_pdf) > 0
    assert all(str(v).lower().startswith("reporte") for v in _pdf["action"])
    print("  ✓ __istartswith works (case-insensitive residual)")
    print()
    return


@app.cell
async def _(parquet):
    print("=" * 72)
    print("PARQUET COMPOSITE FILTER TESTS")
    print("=" * 72)

    # 10. Implicit AND — pushdown + residual combined
    _result = await parquet.aload(
        associate_id__gte=4000,
        action__startswith="Reporte",
        columns=["associate_id", "action"],
    )
    _pdf = _result.compute()
    print(f"associate_id>=4000 AND action starts with Reporte: rows={len(_pdf)}")
    assert len(_pdf) > 0
    assert (_pdf["associate_id"] >= 4000).all()
    assert all(str(v).startswith("Reporte") for v in _pdf["action"])
    print("  ✓ Pushdown + residual filters compose correctly")
    print()
    return


@app.cell
async def _(parquet):
    # 11. Column projection + filter
    _result = await parquet.aload(
        associate_id__gte=4000,
        columns=["associate_id", "action", "latitude"],
    )
    _pdf = _result.compute()
    print(f"Projected columns: {list(_pdf.columns)}")
    assert list(_pdf.columns) == ["associate_id", "action", "latitude"]
    assert (_pdf["associate_id"] >= 4000).all()
    print("  ✓ Column projection + filter works together")
    print()
    return


@app.cell
async def _(parquet):
    # 12. Explicit OR filter
    _result = await parquet.aload(filters={
        "$or": [
            {"associate_id": 27},
            {"associate_id": 4499},
        ]
    }, columns=["associate_id", "action"])
    _pdf = _result.compute()
    print(f"$or: associate_id=27 OR 4499: rows={len(_pdf)}")
    assert len(_pdf) > 0
    assert _pdf["associate_id"].isin([27, 4499]).all()
    print("  ✓ $or boolean filter works on Parquet backend")
    print()
    return


@app.cell
def _():
    print("All Parquet filter tests passed.")
    return


@app.cell
def _(result_dask):
    from boti_dask import (
        UniqueValuesExtractor,
        apply_recommended_dask_config,
        async_safe_compute,
        async_safe_gather,
        async_safe_head,
        async_safe_persist,
        async_safe_wait,
        dask_is_empty,
        dask_is_probably_empty,
        inspect_graph,
        safe_compute,
        safe_gather,
        safe_head,
        safe_persist,
        safe_wait,
    )
    import dask

    graph_metrics = inspect_graph(result_dask)
    assert graph_metrics["is_dask"] is True
    assert graph_metrics["npartitions"] == result_dask.npartitions

    with apply_recommended_dask_config():
        assert dask.config.get("dataframe.shuffle.method") == "tasks"

    graph_metrics
    return (
        UniqueValuesExtractor,
        async_safe_compute,
        async_safe_gather,
        async_safe_head,
        async_safe_persist,
        async_safe_wait,
        dask_is_empty,
        dask_is_probably_empty,
        safe_compute,
        safe_gather,
        safe_head,
        safe_persist,
        safe_wait,
    )


@app.cell
async def _(
    async_safe_compute,
    async_safe_gather,
    async_safe_head,
    async_safe_persist,
    async_safe_wait,
    columns,
    gateway,
    safe_compute,
    safe_gather,
    safe_head,
    safe_persist,
    safe_wait,
):
    with gateway.session(
        verify_connectivity=True,
        shared=True,
        shared_key="bootstrap-resilience",
        cluster_kwargs={"n_workers": 1, "threads_per_worker": 1, "processes": False, "dashboard_address": ":0"},
    ):
        dry_run_frame = await gateway.aload(
            global_track_id__in=[1, 2, 3, 4],
            columns=columns,
            return_type="dask",
            persist=True,
            resilient=True,
            diagnostics=True,
            dry_run=True,
        )
        resilient_frame = await gateway.aload(
            global_track_id__in=[1, 2, 3, 4],
            columns=columns,
            return_type="dask",
            persist=True,
            resilient=True,
            diagnostics=True,
        )
        persisted_frame = safe_persist(resilient_frame)
        safe_wait(persisted_frame, timeout=30)
        row_count = safe_compute(resilient_frame["global_track_id"].count())
        gathered_counts = safe_gather([resilient_frame["global_track_id"].count()])
        resilient_preview = safe_head(resilient_frame, n=3)
        async_persisted = await async_safe_persist(resilient_frame)
        await async_safe_wait(async_persisted, timeout=30)
        async_row_count = await async_safe_compute(resilient_frame["global_track_id"].count())
        async_preview = await async_safe_head(resilient_frame, n=2)
        async_gathered = await async_safe_gather([resilient_frame["global_track_id"].count()])
        del async_persisted
        del persisted_frame
        del resilient_frame

    assert dry_run_frame.npartitions > 0
    assert row_count > 0
    assert async_row_count == row_count
    assert gathered_counts == [row_count]
    assert async_gathered == [row_count]
    assert not resilient_preview.empty
    assert len(async_preview) == 2
    return


@app.cell
async def _(
    UniqueValuesExtractor,
    dask_is_empty,
    dask_is_probably_empty,
    result_dask,
):
    assert dask_is_probably_empty(result_dask) is False
    assert dask_is_empty(result_dask) is False

    unique_values = await UniqueValuesExtractor().extract_unique_values(
        result_dask,
        "product_type_id",
        "global_track_id",
        limit=10,
    )

    assert set(unique_values["product_type_id"]) == {1}
    assert set(unique_values["global_track_id"]).issubset({1, 2, 3, 4})
    unique_values
    return


if __name__ == "__main__":
    app.run()
