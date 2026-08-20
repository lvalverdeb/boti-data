from boti_data._optional import dataframes_required

# Everything re-exported here reaches dask/pandas/polars, so surface the
# missing-extra hint instead of a bare "No module named 'dask'".
with dataframes_required(__name__):
    from .expressions import (
        And,
        ColOp,
        Expr,
        Not,
        Or,
        ParquetFilter,
        ParquetFilterGroup,
        ParquetFilters,
        TrueExpr,
    )
    from .handler import FilterHandler
    from .utils import validate_regex_pattern

__all__ = [
    "And",
    "ColOp",
    "Expr",
    "FilterHandler",
    "Not",
    "Or",
    "ParquetFilter",
    "ParquetFilterGroup",
    "ParquetFilters",
    "TrueExpr",
    "validate_regex_pattern",
]
