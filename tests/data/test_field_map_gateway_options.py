"""
DataFrameParams (chunk_size, index_col), DataFrameOptions (sort_field,
duplicate_expr/keep), exclude=True / use_exclude compat, and
_filter_to_fieldnames post-load column selection.

Split out of test_field_map_gateway.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import pandas as pd

from boti_data.field_map import FieldMap
from boti_data.gateway import DataFrameParams, DataGateway

from .conftest import PRODUCT_MAP, _legacy_gw

# ---------------------------------------------------------------------------
# DataFrameParams: chunk_size, index_col
# DataFrameOptions: sort_field, duplicate_expr/keep
# ---------------------------------------------------------------------------


def test_df_params_index_col(legacy_dsn) -> None:
    """index_col promotes the named column to the DataFrame index."""
    gw = _legacy_gw(
        legacy_dsn,
        df_params=DataFrameParams(index_col="global_track_id"),
    )
    try:
        df = gw.load(as_pandas=True)
        assert df.index.name == "global_track_id"
        assert "global_track_id" not in df.columns
    finally:
        gw.close()


def test_df_params_chunk_size_from_config(legacy_dsn) -> None:
    """chunk_size in DataFrameParams is passed to the partitioned loader."""
    gw = DataGateway.from_config(
        {
            "backend": "sqlalchemy",
            "connection_url": legacy_dsn,
            "table": "legacy_products",
            "field_map": PRODUCT_MAP,
            "df_params": {"chunk_size": 1},
        },
        query_only=False,
    )
    try:
        df = gw.load(as_pandas=True)
        assert len(df) == 3
    finally:
        gw.close()


def test_df_options_sort_field(legacy_dsn) -> None:
    """sort_field produces a DataFrame sorted by that column."""
    from boti_data.gateway import DataFrameOptions

    gw = _legacy_gw(
        legacy_dsn,
        df_options=DataFrameOptions(sort_field="global_track_id"),
    )
    try:
        df = gw.load(as_pandas=True)
        assert list(df["global_track_id"]) == sorted(df["global_track_id"].tolist())
    finally:
        gw.close()


def test_df_options_dedup(legacy_dsn) -> None:
    """duplicate_expr + duplicate_keep drops duplicates by the given column."""
    from boti_data.gateway import DataFrameOptions

    gw = _legacy_gw(
        legacy_dsn,
        sticky_filters={"product_type_id": 1},
        df_options=DataFrameOptions(
            duplicate_expr=["product_type_id"],
            duplicate_keep="last",
        ),
    )
    try:
        df = gw.load(as_pandas=True)
        # product_type_id=1 appears twice — dedup keeps 1 row
        assert len(df) == 1
        assert df.iloc[0]["product_type_id"] == 1
    finally:
        gw.close()


def test_df_options_from_config(legacy_dsn) -> None:
    """df_options dict is accepted by from_config() and applied after load."""
    gw = DataGateway.from_config(
        {
            "backend": "sqlalchemy",
            "connection_url": legacy_dsn,
            "table": "legacy_products",
            "field_map": PRODUCT_MAP,
            "df_options": {
                "sort_field": "global_track_id",
                "duplicate_expr": ["product_type_id"],
                "duplicate_keep": "last",
            },
        },
        query_only=False,
    )
    try:
        df = gw.load(as_pandas=True)
        # 2 distinct product_type_id values → 2 rows after dedup
        assert len(df) == 2
        # Sorted ascending by global_track_id
        assert df.iloc[0]["global_track_id"] < df.iloc[1]["global_track_id"]
    finally:
        gw.close()


# ---------------------------------------------------------------------------
# exclude=True / use_exclude compat
# ---------------------------------------------------------------------------


def test_exclude_negates_sticky_filter(legacy_dsn) -> None:
    """exclude=True returns rows that do NOT match the sticky filter."""
    gw = _legacy_gw(
        legacy_dsn,
        sticky_filters={"product_type_id": 1},
        exclude=True,
    )
    try:
        df = gw.load(as_pandas=True)
        # Only 1 row has product_type_id=2; the two with type=1 are excluded
        assert len(df) == 1
        assert df.iloc[0]["product_type_id"] == 2
    finally:
        gw.close()


def test_exclude_with_runtime_filter(legacy_dsn) -> None:
    """exclude=True with a runtime kwarg excludes the matched row."""
    gw = _legacy_gw(legacy_dsn, exclude=True)
    try:
        df = gw.load(global_track_id=10, as_pandas=True)
        # Row with global_track_id=10 is excluded; 2 rows remain
        assert len(df) == 2
        assert 10 not in df["global_track_id"].tolist()
    finally:
        gw.close()


def test_use_exclude_compat_key(legacy_dsn) -> None:
    """'use_exclude' is accepted as a legacy alias for 'exclude' in from_config."""
    gw = DataGateway.from_config(
        {
            "backend": "sqlalchemy",
            "connection_url": legacy_dsn,
            "table": "legacy_products",
            "field_map": PRODUCT_MAP,
            "use_exclude": True,
            "sticky_filters": {"product_type_id": 1},
        },
        query_only=False,
    )
    try:
        df = gw.load(as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["product_type_id"] == 2
    finally:
        gw.close()


# ===========================================================================
# _filter_to_fieldnames — post-load column selection and missing-col warning
# ===========================================================================


def test_filter_to_fieldnames_sql_selects_subset(legacy_dsn) -> None:
    """fieldnames narrows SQL SELECT; post-load filter is a no-op safety guard."""
    gw = _legacy_gw(
        legacy_dsn,
        df_params=DataFrameParams(fieldnames=("global_track_id", "barcode")),
    )
    try:
        df = gw.load(as_pandas=True)
        assert set(df.columns) == {"global_track_id", "barcode"}
    finally:
        gw.close()


def test_filter_to_fieldnames_missing_col_warns() -> None:
    """Requesting a fieldname absent from the loaded DataFrame emits a UserWarning."""
    import warnings

    from boti_data.gateway import DataFrameOptions
    from boti_data.gateway.post_process import PostProcessor, ResultShapingConfig

    pp = PostProcessor(
        ResultShapingConfig(
            FieldMap({}),
            DataFrameParams(fieldnames=("global_track_id", "nonexistent_col")),
            DataFrameOptions(),
        )
    )
    df = pd.DataFrame({"global_track_id": [10, 20], "barcode": ["A", "B"]})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = pp.filter_to_fieldnames(df)

    assert any("nonexistent_col" in str(warning.message) for warning in w)
    assert "global_track_id" in result.columns
    assert "nonexistent_col" not in result.columns
