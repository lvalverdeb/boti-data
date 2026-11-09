"""Lazy access to the optional dataframe engines.

boti-data's base install carries pandas-free single-row access only; pandas,
polars and Dask arrive through the ``pandas``, ``polars`` and ``dask`` extras.
Modules therefore never import those engines at module level. They import the
accessors here instead, so a pandas-only user never loads (or installs) Dask,
and a missing engine is reported as the extra that supplies it.

Engine detection (`is_dask_frame` and friends) consults ``sys.modules``: an
object can only be a Dask collection if Dask is already imported, so the check
is exact and never triggers an import.
"""

from __future__ import annotations

import sys
from typing import Any

from boti_data._optional import engine_required

__all__ = [
    "boti_dask",
    "dask",
    "dask_dataframe",
    "is_dask_collection",
    "is_dask_frame",
    "is_dask_series",
    "is_pandas_frame",
    "is_pandas_series",
    "is_polars_frame",
    "is_polars_series",
    "polars",
]


def dask_dataframe() -> Any:
    """The ``dask.dataframe`` module (``dask`` extra)."""
    with engine_required("dask"):
        import dask.dataframe as dd
    return dd


def dask() -> Any:
    """The ``dask`` module (``dask`` extra)."""
    with engine_required("dask"):
        import dask as dask_module
    return dask_module


def boti_dask() -> Any:
    """The ``boti_dask`` package (``dask`` extra)."""
    with engine_required("dask"):
        import boti_dask as boti_dask_module
    return boti_dask_module


def polars() -> Any:
    """The ``polars`` module (``polars`` extra)."""
    with engine_required("polars"):
        import polars as polars_module
    return polars_module


def _loaded_class(module_name: str, class_name: str) -> type | None:
    module = sys.modules.get(module_name)
    return getattr(module, class_name, None) if module is not None else None


def _is_instance_of(obj: Any, module_name: str, class_name: str) -> bool:
    cls = _loaded_class(module_name, class_name)
    return cls is not None and isinstance(obj, cls)


def is_dask_frame(obj: Any) -> bool:
    return _is_instance_of(obj, "dask.dataframe", "DataFrame")


def is_dask_series(obj: Any) -> bool:
    return _is_instance_of(obj, "dask.dataframe", "Series")


def is_dask_collection(obj: Any) -> bool:
    return is_dask_frame(obj) or is_dask_series(obj)


def is_polars_frame(obj: Any) -> bool:
    return _is_instance_of(obj, "polars", "DataFrame")


def is_polars_series(obj: Any) -> bool:
    return _is_instance_of(obj, "polars", "Series")


def is_pandas_frame(obj: Any) -> bool:
    return _is_instance_of(obj, "pandas", "DataFrame")


def is_pandas_series(obj: Any) -> bool:
    return _is_instance_of(obj, "pandas", "Series")
