"""Regression coverage for the ``dataframes`` extra's missing-dependency hint.

A base ``pip install boti-data`` deliberately omits dask/pandas/polars/boti-dask
so the SqlRepository API stays lightweight. Reaching for the dataframe side of
the package in that install used to fail with a bare "No module named 'dask'"
that never mentioned the extra; boti_data._optional.dataframes_required now
translates it. These run in a subprocess (same idiom as
test_repository_dependency_free_import) because the check has to happen at
import time, before the real dask/pandas/polars in the dev environment are
already resident in sys.modules.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Blocks the dataframes extra's modules the way a base install would.
_BLOCKER = """
import sys

class _Blocker:
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {"boti_dask", "dask", "pandas", "polars"}:
            raise ModuleNotFoundError(f"No module named {root!r}", name=root)
        return None

sys.meta_path.insert(0, _Blocker())
"""

# Every entry point onto the dataframe side of the package: the lazy re-exports
# on boti_data itself and on boti_data.db, plus each eagerly-importing
# subpackage that consumers import from directly.
DATAFRAME_ENTRY_POINTS = [
    "from boti_data import DataHelper",
    "from boti_data import DataGateway",
    "from boti_data.datacube import BaseDataCube",
    "from boti_data.dataset import HybridDataset",
    "from boti_data.db import SqlPartitionedLoader",
    "from boti_data.enrichment import TypeCaster",
    "from boti_data.filters import FilterHandler",
    "from boti_data.gateway import DataGateway",
    "from boti_data.parquet import ParquetDataConfig",
    "from boti_data.pipelines import ParquetSink",
    "from boti_data.watermark import advance_watermark",
]


def _run_without_extra(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("statement", DATAFRAME_ENTRY_POINTS)
def test_dataframe_entry_point_names_the_extra(statement: str) -> None:
    result = _run_without_extra(
        f"""
        try:
            {statement}
        except ModuleNotFoundError as exc:
            print(exc)
        else:
            raise AssertionError("expected ModuleNotFoundError")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "boti-data[dataframes]" in result.stdout, result.stdout


def test_base_install_surface_still_imports_without_the_extra() -> None:
    """The lightweight SQL API must keep working on a base install -- the guard
    must not turn the whole package into a dataframes-only import."""
    result = _run_without_extra(
        """
        import boti_data
        from boti_data import SqlRepository, AsyncSqlRepository, SqlUnitOfWork
        from boti_data.db import SqlModelRegistry, nearest_neighbors
        print("OK")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_unrelated_import_error_passes_through() -> None:
    """Only the extra's own modules get rewritten; a genuine missing module --
    e.g. a typo inside boti_data -- must surface unchanged."""
    from boti_data._optional import dataframes_required

    with pytest.raises(ModuleNotFoundError, match="definitely_not_a_real_module"):
        with dataframes_required("test"):
            import definitely_not_a_real_module  # noqa: F401

    excinfo_message = ""
    try:
        with dataframes_required("test"):
            raise ModuleNotFoundError("No module named 'nope'", name="nope")
    except ModuleNotFoundError as exc:
        excinfo_message = str(exc)
    assert "boti-data[dataframes]" not in excinfo_message
