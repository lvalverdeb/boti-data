"""
DataGateway example for parquet-backed loading.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from boti_data.gateway import DataGateway


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
        data_dir = project_root / "data"
        data_dir.mkdir()
        pd.DataFrame(
            {"id": [1, 2], "status": ["active", "inactive"], "description": ["urgent", "routine"]}
        ).to_parquet(data_dir / "users.parquet", index=False)

        with DataGateway.from_backend(
            "parquet",
            project_root=project_root,
            logger=StubLogger(),
            parquet_storage_path=str(data_dir),
            parquet_filename="users",
        ) as facade:
            dask_frame = facade.load(filters={"status__exact": "active"})
            pandas_frame = facade.load(
                filters={"status__exact": "active"},
                return_type="pandas",
            )
            arrow_table = facade.load(
                filters={"status__exact": "active"},
                return_type="arrow",
            )
            polars_frame = facade.load(
                filters={"status__exact": "active"},
                return_type="polars",
            )
            auto_frame = facade.load(
                filters={"status__exact": "active"},
                return_type="auto",
            )
            lazy_pandas_frame = facade.load(
                filters={"status__exact": "active"},
                return_type="pandas",
                execution_mode="lazy",
            )

            print("dask")
            print(dask_frame.compute())
            print("\npandas")
            print(pandas_frame)
            print("\narrow")
            print(arrow_table)
            print("\npolars")
            print(polars_frame)
            print("\nauto")
            print(auto_frame)
            print("\npandas over lazy fetch")
            print(lazy_pandas_frame)


if __name__ == "__main__":
    main()
