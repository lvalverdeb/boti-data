"""
Optional-dependency guard for boti-data's ``dataframes`` extra.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

# Top-level modules supplied by [project.optional-dependencies] dataframes:
# boti-dask, dask[dataframe,distributed], pandas and polars. The extra installs
# all four together, so any one of them missing means it was never installed.
_DATAFRAME_MODULES = frozenset({"boti_dask", "dask", "pandas", "polars"})


@contextmanager
def dataframes_required(consumer_name: str) -> Iterator[None]:
    """Turn a missing ``dataframes`` extra into an error that names the extra.

    The dataframe-shaped side of the package (DataGateway/DataHelper, the sink
    pipelines, the datacube/enrichment layers) needs dask, pandas and polars,
    while the lightweight SqlRepository/AsyncSqlRepository API deliberately does
    not -- so a base ``pip install boti-data`` omits them. Without this, reaching
    for the dataframe side raises a bare "No module named 'dask'" that never
    mentions ``boti-data[dataframes]``. Same idea as the missing-DBAPI-driver
    translation in boti_data.db.sql_engine. Every other ModuleNotFoundError --
    including a genuine typo inside boti_data itself -- is re-raised untouched.
    """
    try:
        yield
    except ModuleNotFoundError as exc:
        root = (exc.name or "").split(".", 1)[0]
        if root not in _DATAFRAME_MODULES:
            raise
        raise ModuleNotFoundError(
            f"{consumer_name} requires boti-data's optional 'dataframes' extra, but "
            f"{root!r} is not installed. Install it with "
            "\"pip install 'boti-data[dataframes]'\" (or \"uv add 'boti-data[dataframes]'\").",
            name=exc.name,
            path=exc.path,
        ) from exc
