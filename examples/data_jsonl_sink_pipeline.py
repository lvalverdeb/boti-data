"""SinkPipeline example: materialize a DataHelper load into a partitioned JSONL dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from boti_data import DataHelper, SinkPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_example_shared import _seed_source_events_db  # noqa: E402


def _write_jsonl_pipeline(helper: DataHelper, root: Path) -> Any:
    pipeline = SinkPipeline(
        helper,
        "jsonl",
        sink_config={
            "storage_path": str(root / "source_events_jsonl"),
            "partition_on": ["partition_date"],
            "project_root": root,
        },
        date_field="event_date",
    )
    try:
        return pipeline.write(filters={"status__exact": "active"})
    finally:
        pipeline.close()


def main() -> dict[str, object]:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        db_path = root / "jsonl_pipeline_example.db"
        worker_dsn_env_var = "BOTI_EXAMPLE_JSONL_SQLITE_DSN"
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
            result = _write_jsonl_pipeline(helper, root)
        finally:
            if previous_worker_dsn is None:
                os.environ.pop(worker_dsn_env_var, None)
            else:
                os.environ[worker_dsn_env_var] = previous_worker_dsn

    summary = {
        "path": result.path,
        "n_files": len(result.files),
        "sample_file": result.files[0] if result.files else None,
    }
    print(f"SinkPipeline wrote JSONL dataset to {summary['path']}")
    print(f"jsonl files={summary['n_files']} sample={summary['sample_file']}")
    return summary


if __name__ == "__main__":
    main()
