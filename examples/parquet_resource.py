"""
ParquetDataResource example covering file and partitioned loads.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from boti_data.parquet import ParquetDataConfig, ParquetDataResource


class StubLogger:
    def debug(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def error(self, *_args, **_kwargs) -> None:
        pass


def main() -> None:
    with TemporaryDirectory() as tmp_dir:
        project_root = Path(tmp_dir)
        (project_root / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")

        single_dir = project_root / "parquet" / "single"
        single_dir.mkdir(parents=True)
        pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).to_parquet(
            single_dir / "users.parquet",
            index=False,
        )

        single_config = ParquetDataConfig(
            project_root=project_root,
            logger=StubLogger(),
            parquet_storage_path=str(single_dir),
            parquet_filename="users",
        )

        with ParquetDataResource(single_config) as resource:
            print(resource.load_files().compute().sort_values("id").reset_index(drop=True))

        partition_root = project_root / "parquet" / "partitioned"
        (partition_root / "2024" / "1" / "1").mkdir(parents=True)
        (partition_root / "2025" / "1" / "1").mkdir(parents=True)
        pd.DataFrame({"id": [10]}).to_parquet(partition_root / "2024" / "1" / "1" / "part_2024.parquet", index=False)
        pd.DataFrame({"id": [20]}).to_parquet(partition_root / "2025" / "1" / "1" / "part_2025.parquet", index=False)

        partition_config = ParquetDataConfig(
            project_root=project_root,
            logger=StubLogger(),
            parquet_storage_path=str(partition_root),
            parquet_start_date=dt.date(2024, 1, 1),
            parquet_end_date=dt.date(2024, 12, 31),
        )

        with ParquetDataResource(partition_config) as resource:
            print(resource.load_files().compute().sort_values("id").reset_index(drop=True))


if __name__ == "__main__":
    main()
