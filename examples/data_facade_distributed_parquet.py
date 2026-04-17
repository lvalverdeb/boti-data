"""
DataGateway example for parquet-backed distributed loading and joins.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from dask.distributed import LocalCluster

from boti_dask import dask_session, describe_client
from boti_data.gateway import DataGateway
from boti_data.joins import indexed_left_join


def run_example() -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        project_root = Path(tmp_dir)
        (project_root / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")
        data_dir = project_root / "warehouse"
        data_dir.mkdir()

        pd.DataFrame(
            {
                "id": pd.Series([1, 2, 3, 4], dtype="Int64"),
                "status": ["active", "inactive", "active", "active"],
            }
        ).to_parquet(data_dir / "users.parquet", index=False)
        pd.DataFrame(
            {
                "id": pd.Series([1, 3, 4], dtype="Int64"),
                "tier": ["gold", "standard", "gold"],
                "region": ["region-01", "region-03", "region-04"],
            }
        ).to_parquet(data_dir / "profiles.parquet", index=False)

        with LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=False,
            dashboard_address=None,
        ) as cluster:
            with dask_session(scheduler_address=cluster.scheduler_address) as client:
                client_summary = describe_client(client)
                with DataGateway.from_backend(
                    "parquet",
                    project_root=project_root,
                    parquet_storage_path=str(data_dir),
                    parquet_filename="users",
                ) as user_gateway, DataGateway.from_backend(
                    "parquet",
                    project_root=project_root,
                    parquet_storage_path=str(data_dir),
                    parquet_filename="profiles",
                ) as profile_gateway:
                    users = user_gateway.load(diagnostics=True)
                    profiles = profile_gateway.load(diagnostics=True)
                    joined = indexed_left_join(
                        users,
                        profiles,
                        join_key="id",
                        join_schema_map={"id": "Int64"},
                        persist=True,
                        diagnostics=True,
                    )
                    head = joined.head(4)
                    matched_rows = int(joined["tier"].count().compute())
                    unmatched_rows = int(joined["tier"].isna().sum().compute())

    return {
        "client": client_summary,
        "head": head,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
    }


def main() -> dict[str, object]:
    result = run_example()
    print("Scheduler session client:")
    print(result["client"])
    print("\nJoined parquet head:")
    print(result["head"])
    print("\nMatch summary:")
    print(f"matched rows: {result['matched_rows']}")
    print(f"unmatched rows: {result['unmatched_rows']}")
    return result


if __name__ == "__main__":
    main()
