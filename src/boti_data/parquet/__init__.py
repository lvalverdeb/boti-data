"""
Parquet-backed data resources.
"""

from __future__ import annotations

import importlib
from typing import Any

from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource

__all__ = ["ParquetDataConfig", "ParquetDataResource", "ParquetReader"]

# ParquetReader is deferred (PEP 562): it depends on boti_data.gateway, which in
# turn depends on this package's own ParquetDataConfig/ParquetDataResource (for
# gateway.requests's BackendConfig/BackendResource unions) — importing it eagerly
# here would make boti_data.parquet.resource unreachable without first fully
# resolving boti_data.gateway, a genuine package-level circular import.
_LAZY = {"ParquetReader": ("boti_data.parquet.reader", "ParquetReader")}


# Deliberately identical to the __getattr__ in boti_data/__init__.py and
# boti_data/db/__init__.py -- this is PEP 562's own documented module-level
# lazy-import boilerplate. Each copy needs its own private
# _LAZY dict/globals()/__name__, so sharing an implementation would need an
# awkward globals()-passing helper for no real gain over these 6 lines.
# spaghetti-ignore[duplicate-function-body]: see above
def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
