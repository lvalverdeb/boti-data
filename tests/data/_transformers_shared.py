"""Shared sample-frame builder for the transformers tests.

Split out purely to deduplicate an identical helper that was copied across
test_transformers_casting_and_filter.py and test_transformers_derived_and_composite.py.
Leading underscore so pytest does not collect this as a test module.
"""

from __future__ import annotations

import pandas as pd


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
            "value": ["10", "20", "30", "40", "50"],
            "active": [1, 0, 1, 0, 1],
            "category": ["x", "y", "x", "y", "x"],
        }
    )
