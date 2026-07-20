"""
PyArrow compute kernels for filter operations.

Provides vectorized, zero-copy equivalents for all pandas/Dask filter
operations using ``pyarrow.compute``. These kernels are 5-50x faster
for string operations and enable $or/$not pushdown that was previously
excluded from parquet pushdown.

Implementation lives in arrow_kernels_comparison.py (type coercion +
comparison kernels), arrow_kernels_string.py (string kernels), and
arrow_kernels_compile.py (boolean combinators, operation dispatch, and the
filter-expression compiler), split out purely for god-module headroom.
Re-exported here so every existing ``from boti_data.filters.arrow_kernels
import ...`` keeps working, including ``handler.py``'s module-attribute
access to ``arrow_kernels.ARROW_PUSHDOWN_OPS``/``arrow_kernels.apply_arrow_filters``.
"""

from __future__ import annotations

from boti_data.filters.arrow_kernels_comparison import (
    ensure_chunked,
    ensure_string_array,
    exact_kernel,
    gt_kernel,
    gte_kernel,
    in_kernel,
    isnull_kernel,
    lt_kernel,
    lte_kernel,
    not_exact_kernel,
    not_in_kernel,
    range_kernel,
)
from boti_data.filters.arrow_kernels_compile import (
    ARROW_OPERATION_KERNELS,
    ARROW_PUSHDOWN_OPS,
    ARROW_RESIDUAL_OPS,
    and_kernel,
    apply_arrow_filters,
    apply_arrow_operation,
    apply_boolean_operators,
    compile_arrow_filter,
    not_kernel,
    or_kernel,
)
from boti_data.filters.arrow_kernels_string import (
    contains_kernel,
    endswith_kernel,
    icontains_kernel,
    iendswith_kernel,
    iexact_kernel,
    istartswith_kernel,
    not_contains_kernel,
    regex_kernel,
    startswith_kernel,
)

__all__ = [
    "ARROW_OPERATION_KERNELS",
    "ARROW_PUSHDOWN_OPS",
    "ARROW_RESIDUAL_OPS",
    "and_kernel",
    "apply_arrow_filters",
    "apply_arrow_operation",
    "apply_boolean_operators",
    "compile_arrow_filter",
    "contains_kernel",
    "endswith_kernel",
    "ensure_chunked",
    "ensure_string_array",
    "exact_kernel",
    "gt_kernel",
    "gte_kernel",
    "icontains_kernel",
    "iendswith_kernel",
    "iexact_kernel",
    "in_kernel",
    "isnull_kernel",
    "istartswith_kernel",
    "lt_kernel",
    "lte_kernel",
    "not_contains_kernel",
    "not_exact_kernel",
    "not_in_kernel",
    "not_kernel",
    "or_kernel",
    "range_kernel",
    "regex_kernel",
    "startswith_kernel",
]
