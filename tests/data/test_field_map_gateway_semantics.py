"""
DataGateway configured mode: field_map present vs. absent semantics, and the
from_config() classmethod.

Semantics under test
--------------------
field_map present  → DB uses non-semantic column names (e.g. 'id_tipo_producto').
                     All inputs (sticky_filters, runtime kwargs, fieldnames) are
                     ALWAYS expressed as semantic names.  The gateway translates
                     them to DB column names before issuing the query, and renames
                     the result columns back to semantic names.

field_map absent   → DB already uses semantic column names.  No translation or
                     rename is performed.

Split out of test_field_map_gateway.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from boti_data.gateway import DataFrameParams, DataGateway

from .conftest import PRODUCT_MAP, _global_track_ids, _legacy_gw, _semantic_gw

# ---------------------------------------------------------------------------
# field_map present  (DB has non-semantic column names → translate)
# ---------------------------------------------------------------------------


def test_legacy_sticky_filter_uses_semantic_names(legacy_dsn) -> None:
    """sticky_filters expressed as semantic names; gateway translates to DB col."""
    gw = _legacy_gw(legacy_dsn, sticky_filters={"product_type_id": 1})
    try:
        df = gw.load(as_pandas=True)
        assert len(df) == 2
        assert set(df["product_type_id"].tolist()) == {1}
        # DB column names must NOT appear in the result
        assert "id_tipo_produto" not in df.columns
    finally:
        gw.close()


def test_legacy_runtime_filter_uses_semantic_names(legacy_dsn) -> None:
    """Runtime aload() kwargs are semantic names."""
    gw = _legacy_gw(legacy_dsn, sticky_filters={"product_type_id": 1})
    try:
        df = gw.load(global_track_id=10, as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["global_track_id"] == 10
    finally:
        gw.close()


def test_legacy_output_columns_are_semantic(legacy_dsn) -> None:
    """Every mapped DB column is renamed to its semantic name in output."""
    gw = _legacy_gw(legacy_dsn)
    try:
        df = gw.load(as_pandas=True)
        for db_col in PRODUCT_MAP:
            assert db_col not in df.columns
        for sem_col in PRODUCT_MAP.values():
            assert sem_col in df.columns
    finally:
        gw.close()


def test_legacy_gateway_arrow_output_uses_semantic_columns(legacy_dsn) -> None:
    gw = _legacy_gw(legacy_dsn, df_params=DataFrameParams(return_type="arrow"))
    try:
        table = gw.load()
    finally:
        gw.close()

    assert isinstance(table, pa.Table)
    assert "id_tipo_produto" not in table.column_names
    assert "product_type_id" in table.column_names


@pytest.mark.parametrize(
    ("return_type", "expected_type"),
    [
        ("dask", dd.DataFrame),
        ("pandas", pd.DataFrame),
        ("arrow", pa.Table),
        ("polars", pl.DataFrame),
        ("auto", pd.DataFrame),
    ],
)
def test_legacy_gateway_return_type_matrix(legacy_dsn, return_type, expected_type) -> None:
    gw = _legacy_gw(legacy_dsn)
    try:
        result = gw.load(return_type=return_type)
    finally:
        gw.close()

    assert isinstance(result, expected_type)
    assert _global_track_ids(result) == [10, 20, 30]

    if isinstance(result, pa.Table):
        assert "id_tipo_produto" not in result.column_names
        assert "product_type_id" in result.column_names
    else:
        assert "id_tipo_produto" not in result.columns
        assert "product_type_id" in result.columns


def test_legacy_fieldnames_are_semantic(legacy_dsn) -> None:
    """fieldnames are semantic → gateway selects only the corresponding DB cols."""
    gw = _legacy_gw(
        legacy_dsn,
        sticky_filters={"product_type_id": 1},
        df_params=DataFrameParams(fieldnames=("global_track_id", "process_track_id")),
    )
    try:
        df = gw.load(as_pandas=True)
        assert set(df.columns) == {"global_track_id", "process_track_id"}
    finally:
        gw.close()


def test_legacy_column_names_positional_override(legacy_dsn) -> None:
    """column_names applies positional rename after the semantic rename."""
    gw = _legacy_gw(
        legacy_dsn,
        sticky_filters={"product_type_id": 1},
        df_params=DataFrameParams(
            fieldnames=("global_track_id", "process_track_id"),
            column_names=["gtrack", "ptrack_x"],
        ),
    )
    try:
        df = gw.load(as_pandas=True)
        assert list(df.columns) == ["gtrack", "ptrack_x"]
    finally:
        gw.close()


def test_legacy_or_filter_with_semantic_names(legacy_dsn) -> None:
    """$or filter dict with semantic names is correctly translated."""
    gw = _legacy_gw(legacy_dsn)
    try:
        df = gw.load(
            filters={
                "$or": [
                    {"product_type_id": 1},
                    {"global_track_id": 30},
                ]
            },
            as_pandas=True,
        )
        # All 3 rows: type=1 (×2) + type=2/track=30 (×1)
        assert len(df) == 3
    finally:
        gw.close()


# ---------------------------------------------------------------------------
# field_map absent  (DB already uses semantic column names → passthrough)
# ---------------------------------------------------------------------------


def test_semantic_sticky_filter(semantic_dsn) -> None:
    """When no field_map is provided no translation or rename occurs."""
    gw = _semantic_gw(semantic_dsn, sticky_filters={"product_type_id": 1})
    try:
        df = gw.load(as_pandas=True)
        assert len(df) == 2
        assert "product_type_id" in df.columns
    finally:
        gw.close()


def test_semantic_runtime_filter(semantic_dsn) -> None:
    gw = _semantic_gw(semantic_dsn, sticky_filters={"product_type_id": 1})
    try:
        df = gw.load(global_track_id=10, as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["global_track_id"] == 10
    finally:
        gw.close()


def test_semantic_no_rename_applied(semantic_dsn) -> None:
    """Without a field_map the DB column names pass through unchanged."""
    gw = _semantic_gw(semantic_dsn)
    try:
        df = gw.load(as_pandas=True)
        # Semantic names are present as-is; no legacy DB names in sight
        assert "product_type_id" in df.columns
        assert "id_tipo_produto" not in df.columns
    finally:
        gw.close()


# ---------------------------------------------------------------------------
# from_config() classmethod
# ---------------------------------------------------------------------------


def test_from_config_mirrors_dfhelper(legacy_dsn) -> None:
    """from_config() accepts the legacy DfHelper dict using field_map-driven translation."""
    gw = DataGateway.from_config(
        {
            "backend": "sqlalchemy",
            "connection_url": legacy_dsn,
            "table": "legacy_products",
            "field_map": PRODUCT_MAP,
            "sticky_filters": {"product_type_id": 2},  # semantic name
            "df_params": {
                "fieldnames": ("global_track_id",),  # semantic name
            },
        },
        query_only=False,
    )
    try:
        df = gw.load(as_pandas=True)
        assert len(df) == 1
        assert "global_track_id" in df.columns
        assert df.iloc[0]["global_track_id"] == 30
    finally:
        gw.close()


def test_from_config_preserves_embedded_fs_when_override_is_none(tmp_path) -> None:
    """A blind cfg.update(overrides) must not let an incidental fs=None from a
    wrapper (e.g. ParquetReader forwarding fs=None when the caller never set it)
    clobber a real fs embedded directly in the config mapping."""
    import fsspec

    real_fs = fsspec.filesystem("memory")
    gw = DataGateway.from_config(
        {
            "backend": "parquet",
            "parquet_storage_path": str(tmp_path / "data"),
            "fs": real_fs,
        },
        fs=None,
    )
    try:
        assert gw.resource.fs is real_fs
    finally:
        gw.close()


def test_from_config_sqlalchemy_default_backend_error_names_the_fallback() -> None:
    """Omitting 'backend' silently defaults to sqlalchemy; the connection_url
    error should say so instead of looking like an unrelated validation bug."""
    with pytest.raises(ValueError, match="default used when no 'backend' key"):
        DataGateway.from_config({"parquet_storage_path": "/tmp/whatever"})
