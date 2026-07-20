from __future__ import annotations

import pyarrow as pa

from boti_data.gateway.arrow_adapters import drop_duplicates


def test_arrow_drop_duplicates_keeps_first_without_python_key_materialization_regression() -> None:
    table = pa.table(
        {
            "id": [1, 1, 2, 2],
            "group": ["a", "a", "b", "b"],
            "value": ["first-a", "second-a", "first-b", "second-b"],
        }
    )

    result = drop_duplicates(table, subset=["id", "group"], keep="first")

    assert result["value"].to_pylist() == ["first-a", "first-b"]


def test_arrow_drop_duplicates_keeps_last_without_python_key_materialization_regression() -> None:
    table = pa.table(
        {
            "id": [1, 1, 2, 2],
            "group": ["a", "a", "b", "b"],
            "value": ["first-a", "second-a", "first-b", "second-b"],
        }
    )

    result = drop_duplicates(table, subset=["id", "group"], keep="last")

    assert result["value"].to_pylist() == ["second-a", "second-b"]
