"""
Bootstrap migration example based on the legacy notebook flow.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from boti_data import DataHelper
from boti_dask import (
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


class Base(DeclarativeBase):
    pass


class TrackingProduct(Base):
    __tablename__ = "asm_tracking_productos"

    id: Mapped[int] = mapped_column(primary_key=True)
    id_producto: Mapped[int]
    cliente_id: Mapped[int]
    id_track_global: Mapped[int]
    id_tipo_producto: Mapped[int]


def _seed_database(db_path: Path) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add_all(
                [
                    TrackingProduct(
                        id_producto=101,
                        cliente_id=11,
                        id_track_global=1,
                        id_tipo_producto=1,
                    ),
                    TrackingProduct(
                        id_producto=102,
                        cliente_id=12,
                        id_track_global=2,
                        id_tipo_producto=1,
                    ),
                    TrackingProduct(
                        id_producto=103,
                        cliente_id=13,
                        id_track_global=3,
                        id_tipo_producto=1,
                    ),
                    TrackingProduct(
                        id_producto=104,
                        cliente_id=14,
                        id_track_global=4,
                        id_tipo_producto=1,
                    ),
                    TrackingProduct(
                        id_producto=999,
                        cliente_id=99,
                        id_track_global=1,
                        id_tipo_producto=2,
                    ),
                ]
            )
            session.commit()
    finally:
        engine.dispose()


async def _run_engine_views(
    helper: DataHelper,
    *,
    columns: list[str],
) -> dict[str, Any]:
    pandas_frame = await helper.pandas.aload(global_track_id__in=[1, 2, 3, 4], columns=columns)
    polars_frame = await helper.polars.aload(global_track_id__in=[1, 2, 3, 4], columns=columns)
    dask_frame = await helper.dask.aload(global_track_id__in=[1, 2, 3, 4], columns=columns)
    return {
        "pandas_rows": int(len(pandas_frame)),
        "polars_rows": int(polars_frame.height),
        "dask_rows": int(dask_frame.shape[0].compute()),
        "pandas_records": pandas_frame.sort_values("global_track_id").to_dict(orient="records"),
        "polars_records": polars_frame.sort("global_track_id").to_dicts(),
    }


async def _run_async_resilience(frame, *, client) -> dict[str, Any]:
    async_persisted = await async_safe_persist(frame, dask_client=client)
    await async_safe_wait(async_persisted, dask_client=client, timeout=30)
    row_count = int(await async_safe_compute(frame["global_track_id"].count(), dask_client=client))
    preview = await async_safe_head(frame, n=2, dask_client=client)
    gathered = await async_safe_gather([frame["global_track_id"].count()], dask_client=client)
    del async_persisted
    return {
        "row_count": row_count,
        "preview_ids": preview["global_track_id"].tolist(),
        "gathered": [int(value) for value in gathered],
    }


def run_example() -> dict[str, Any]:
    with TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "bootstrap_legacy.db"
        _seed_database(db_path)

        helper = DataHelper(
            backend="sqlalchemy",
            connection_url=f"sqlite:///{db_path}",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
            table="asm_tracking_productos",
            field_map={
                "id_track_global": "global_track_id",
                "id_tipo_producto": "product_type_id",
            },
            sticky_filters={"product_type_id": 1},
        )

        columns = ["id_producto", "cliente_id", "product_type_id", "global_track_id"]
        try:
            with helper.session(
                verify_connectivity=True,
                shared=True,
                shared_key="bootstrap-legacy-migration",
                cluster_kwargs={
                    "n_workers": 1,
                    "threads_per_worker": 1,
                    "processes": False,
                    "dashboard_address": ":0",
                },
            ) as client:
                engine_views = asyncio.run(_run_engine_views(helper, columns=columns))
                dry_run_frame = asyncio.run(
                    helper.aload(
                        global_track_id__in=[1, 2, 3, 4],
                        columns=columns,
                        return_type="dask",
                        persist=True,
                        resilient=True,
                        diagnostics=True,
                        dry_run=True,
                    )
                )
                resilient_frame = asyncio.run(
                    helper.aload(
                        global_track_id__in=[1, 2, 3, 4],
                        columns=columns,
                        return_type="dask",
                        persist=True,
                        resilient=True,
                        diagnostics=True,
                    )
                )
                persisted_frame = safe_persist(resilient_frame, dask_client=client)
                safe_wait(persisted_frame, dask_client=client, timeout=30)
                sync_row_count = int(
                    safe_compute(resilient_frame["global_track_id"].count(), dask_client=client)
                )
                sync_gather = [
                    int(value)
                    for value in safe_gather(
                        [resilient_frame["global_track_id"].count()],
                        dask_client=client,
                    )
                ]
                sync_preview = safe_head(resilient_frame, n=3, dask_client=client)
                async_summary = asyncio.run(_run_async_resilience(resilient_frame, client=client))
                probably_empty = dask_is_probably_empty(resilient_frame)
                is_empty = dask_is_empty(resilient_frame, dask_client=client)
                unique_values = asyncio.run(
                    UniqueValuesExtractor(dask_client=client).extract_unique_values(
                        resilient_frame,
                        "product_type_id",
                        "global_track_id",
                        limit=10,
                    )
                )
                del persisted_frame
                del resilient_frame
        finally:
            helper.close()

    return {
        "engine_views": engine_views,
        "dry_run_partitions": int(dry_run_frame.npartitions),
        "sync_row_count": sync_row_count,
        "sync_gather": sync_gather,
        "sync_preview_ids": sync_preview["global_track_id"].tolist(),
        "async_row_count": async_summary["row_count"],
        "async_preview_ids": async_summary["preview_ids"],
        "async_gather": async_summary["gathered"],
        "probably_empty": probably_empty,
        "is_empty": is_empty,
        "unique_values": unique_values,
    }


def main() -> dict[str, Any]:
    result = run_example()
    print("Bootstrap engine-view rows:")
    print(result["engine_views"])
    print(f"Dry-run partitions: {result['dry_run_partitions']}")
    print(f"Sync row count: {result['sync_row_count']}")
    print(f"Sync preview ids: {result['sync_preview_ids']}")
    print(f"Sync gathered values: {result['sync_gather']}")
    print(f"Async row count: {result['async_row_count']}")
    print(f"Async preview ids: {result['async_preview_ids']}")
    print(f"Async gathered values: {result['async_gather']}")
    print(f"Probably empty: {result['probably_empty']}")
    print(f"Is empty: {result['is_empty']}")
    print(f"Unique values: {result['unique_values']}")
    return result


if __name__ == "__main__":
    main()

