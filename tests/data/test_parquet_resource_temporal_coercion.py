"""Tz-aware timestamp-column filter coercion tests for ParquetDataResource.

Split out of test_parquet_resource_loading.py purely for long-file headroom
(mirrors that file's own string-column coercion tests, which stay in place).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from boti_data import DataHelper, ParquetDataConfig, ParquetDataResource

from .test_parquet_resource_loading import StubLogger


def test_parquet_coerces_bare_date_filter_on_tz_aware_timestamp_column(temp_project_root) -> None:
    """Regression: a bare ``datetime.date`` filter value against a genuine
    tz-aware timestamp column is promoted to a matching ``pd.Timestamp``
    instead of raising ``ArrowInvalid``.

    ``prepare_period_filters()`` (``gateway/normalization.py``) always
    truncates its bounds to a bare ``date`` regardless of the target
    column's actual dtype. PyArrow has no comparison kernel between
    ``date32`` and a tz-aware ``timestamp`` column, so this previously
    raised unconditionally for any tz-aware ``date_field``. Filtering the
    identical column with an actual ``pd.Timestamp`` already worked; this
    closes the bare-``date`` gap symmetrically to the string-column case in
    test_parquet_resource_loading.py.
    """
    file_path = temp_project_root / "data" / "events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "last_activity_dt": pd.to_datetime(
                ["2026-04-15", "2026-04-17", "2026-04-25"], utc=True
            ),
        }
    ).to_parquet(file_path, index=False)

    config = ParquetDataConfig(
        project_root=temp_project_root,
        logger=StubLogger(),
        parquet_storage_path=str(file_path.parent),
        parquet_filename="events",
    )

    with ParquetDataResource(config) as resource:
        loaded = (
            resource.load_filtered({"last_activity_dt__gte": dt.date(2026, 4, 17)})
            .compute()
            .sort_values("id")
        )
        assert loaded["id"].tolist() == [2, 3]

        table = resource.load_filtered_arrow({"last_activity_dt__lt": dt.date(2026, 4, 17)})
        assert table.column("id").to_pylist() == [1]


def test_datahelper_parquet_load_period_on_tz_aware_timestamp_column(temp_project_root) -> None:
    """Same regression as above, exercised end-to-end via ``load_period`` —
    the exact path ``HybridDataset``'s parquet branch uses."""
    file_path = temp_project_root / "data" / "events.parquet"
    file_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "last_activity_dt": pd.to_datetime(
                ["2026-04-15", "2026-04-17", "2026-04-25"], utc=True
            ),
        }
    ).to_parquet(file_path, index=False)

    helper = DataHelper(
        backend="parquet",
        storage_path=str(file_path.parent),
        parquet_filename="events",
    )
    try:
        window = helper.load_period(
            "last_activity_dt", "2026-04-15", "2026-04-19", return_type="pandas"
        )
    finally:
        helper.close()

    assert sorted(window["id"].tolist()) == [1, 2]
