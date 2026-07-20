"""ParquetPipeline example: materialize a DataHelper load and optionally reload parquet."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from boti_data import DataHelper, ParquetPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_example_shared import _seed_source_events_db  # noqa: E402


def _materialize_parquet_pipeline(helper: DataHelper, root: Path) -> dict[str, object]:
    pipeline = ParquetPipeline(
        helper,
        {
            "backend": "parquet",
            "storage_path": str(root / "source_events_dataset"),
            "project_root": root,
            "partition_on": ["partition_date"],
        },
        date_field="event_date",
    )
    try:
        write_only = pipeline.materialize(filters={"status__exact": "active"})
        reloaded = pipeline.materialize(
            filters={"status__exact": "active"},
            reload=True,
            reload_options={"filters": {"partition_date__exact": "2026-04-17"}},
        )
        reloaded_rows = int(reloaded.frame.compute().shape[0]) if reloaded.frame is not None else 0
    finally:
        pipeline.close()
    return {
        "path": write_only.path,
        "write_only_reloaded": write_only.reloaded,
        "reload_reloaded": reloaded.reloaded,
        "reload_rows": reloaded_rows,
    }


def main() -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        db_path = root / "pipeline_example.db"
        worker_dsn_env_var = "BOTI_EXAMPLE_SQLITE_DSN"
        sqlite_dsn = f"sqlite:///{db_path}"
        previous_worker_dsn = os.environ.get(worker_dsn_env_var)
        os.environ[worker_dsn_env_var] = sqlite_dsn
        _seed_source_events_db(sqlite_dsn)

        helper = DataHelper(
            backend="sqlalchemy",
            connection_url=sqlite_dsn,
            worker_connection_env_var=worker_dsn_env_var,
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
            table="source_events",
        )
        try:
            result = _materialize_parquet_pipeline(helper, root)
        finally:
            if previous_worker_dsn is None:
                os.environ.pop(worker_dsn_env_var, None)
            else:
                os.environ[worker_dsn_env_var] = previous_worker_dsn

    print(f"ParquetPipeline wrote dataset to {result['path']}")
    print(f"write_only.reloaded={result['write_only_reloaded']}")
    print(f"reloaded.reloaded={result['reload_reloaded']} rows={result['reload_rows']}")
    return result


if __name__ == "__main__":
    main()
