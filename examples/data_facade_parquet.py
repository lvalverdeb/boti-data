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


def _seed_users_parquet(data_dir: Path) -> None:
    pd.DataFrame(
        {"id": [1, 2], "status": ["active", "inactive"], "description": ["urgent", "routine"]}
    ).to_parquet(data_dir / "users.parquet", index=False)


def _load_all_frames(facade: DataGateway) -> dict[str, object]:
    active = {"status__exact": "active"}
    return {
        "dask": facade.load(filters=active),
        "pandas": facade.load(filters=active, return_type="pandas"),
        "arrow": facade.load(filters=active, return_type="arrow"),
        "polars": facade.load(filters=active, return_type="polars"),
        "auto": facade.load(filters=active, return_type="auto"),
        "lazy_pandas": facade.load(filters=active, return_type="pandas", execution_mode="lazy"),
    }


def _print_frames(frames: dict[str, object]) -> None:
    print("dask")
    print(frames["dask"].compute())
    print("\npandas")
    print(frames["pandas"])
    print("\narrow")
    print(frames["arrow"])
    print("\npolars")
    print(frames["polars"])
    print("\nauto")
    print(frames["auto"])
    print("\npandas over lazy fetch")
    print(frames["lazy_pandas"])


def main() -> None:
    with TemporaryDirectory() as tmp_dir:
        project_root = Path(tmp_dir)
        (project_root / "pyproject.toml").write_text(
            "[project]\nname='example'\n", encoding="utf-8"
        )
        data_dir = project_root / "data"
        data_dir.mkdir()
        _seed_users_parquet(data_dir)

        with DataGateway.from_backend(
            "parquet",
            project_root=project_root,
            logger=StubLogger(),
            parquet_storage_path=str(data_dir),
            parquet_filename="users",
        ) as facade:
            frames = _load_all_frames(facade)
            _print_frames(frames)


if __name__ == "__main__":
    main()
