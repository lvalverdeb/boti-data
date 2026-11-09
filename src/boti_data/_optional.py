"""
Optional-dependency guards for boti-data's engine extras.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

# Top-level modules each extra supplies. ``dataframes`` is the union of the
# three and stays available for users who relied on it pulling everything.
_ENGINE_MODULES: dict[str, frozenset[str]] = {
    "pandas": frozenset({"pandas"}),
    "polars": frozenset({"polars"}),
    "dask": frozenset({"dask", "distributed", "boti_dask"}),
}
_DATAFRAME_MODULES = frozenset().union(*_ENGINE_MODULES.values())


def _extra_for(root: str) -> str | None:
    for extra, modules in _ENGINE_MODULES.items():
        if root in modules:
            return extra
    return None


@contextmanager
def engine_required(engine: str, consumer_name: str | None = None) -> Iterator[None]:
    """Turn a missing engine import into an error that names the extra supplying it.

    Every other ModuleNotFoundError, including a genuine typo inside boti_data,
    is re-raised untouched. Same idea as the missing-DBAPI-driver translation in
    boti_data.db.sql_engine.
    """
    try:
        yield
    except ModuleNotFoundError as exc:
        root = (exc.name or "").split(".", 1)[0]
        extra = _extra_for(root)
        if extra is None:
            raise
        who = consumer_name or f"boti-data's {engine} support"
        raise ModuleNotFoundError(
            f"{who} requires boti-data's optional '{extra}' extra, but {root!r} is not "
            f"installed. Install it with \"pip install 'boti-data[{extra}]'\" "
            f"(or \"uv add 'boti-data[{extra}]'\"); 'boti-data[dataframes]' installs every engine.",
            name=exc.name,
            path=exc.path,
        ) from exc


@contextmanager
def dataframes_required(consumer_name: str) -> Iterator[None]:
    """Guard an import that may need any engine; the error names the specific extra."""
    with engine_required("dataframes", consumer_name):
        yield
