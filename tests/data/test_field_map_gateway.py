"""
Tests for FieldMap translation and DataGateway configured mode.

Semantics under test
--------------------
field_map present  → DB uses non-semantic column names (e.g. 'id_tipo_producto').
                     All inputs (sticky_filters, runtime kwargs, fieldnames) are
                     ALWAYS expressed as semantic names.  The gateway translates
                     them to DB column names before issuing the query, and renames
                     the result columns back to semantic names.

field_map absent   → DB already uses semantic column names.  No translation or
                     rename is performed.

Parquet            → Files already carry semantic column names; translation and
                     rename are never applied.
"""

from __future__ import annotations

from typing import Any

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from pydantic import SecretStr
from sqlalchemy import String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

import boti_data.gateway._series_filters as gateway_series_filters
import boti_data.gateway.core as gateway_core
import boti_data.gateway.return_type as gateway_return_type
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.field_map import FieldMap
from boti_data.gateway import DataFrameParams, DataGateway, GatewayPolicies
from boti_data.gateway.loaders import load_sql, load_sql_partitioned
from boti_data.gateway.planning import InChunkPolicy
from boti_data.gateway.requests import SqlLoadRequest

# ---------------------------------------------------------------------------
# Shared field map (DB legacy name → semantic name)
# ---------------------------------------------------------------------------

PRODUCT_MAP: dict[str, str] = {
    "id_tipo_produto": "product_type_id",
    "id_track_global": "global_track_id",
    "id_track_proceso": "process_track_id",
    "codigo_barra": "barcode",
    "nombre_th": "first_name",
}


@pytest.fixture
def fmap():
    return FieldMap(PRODUCT_MAP)


# ===========================================================================
# FieldMap unit tests
# ===========================================================================


def test_fieldmap_to_db(fmap):
    assert fmap.to_db("product_type_id") == "id_tipo_produto"
    assert fmap.to_db("global_track_id") == "id_track_global"
    # Unknown semantic name passes through unchanged
    assert fmap.to_db("unknown_col") == "unknown_col"


def test_fieldmap_to_semantic(fmap):
    assert fmap.to_semantic("id_tipo_produto") == "product_type_id"
    # Unknown DB column passes through unchanged
    assert fmap.to_semantic("some_other_col") == "some_other_col"


def test_fieldmap_bool():
    assert bool(FieldMap(PRODUCT_MAP)) is True
    assert bool(FieldMap({})) is False


def test_fieldmap_duplicate_semantic_raises():
    with pytest.raises(ValueError, match="Duplicate semantic name"):
        FieldMap({"col_a": "same", "col_b": "same"})


def test_fieldmap_select_db_columns(fmap):
    result = fmap.select_db_columns(["product_type_id", "global_track_id"])
    assert result == ["id_tipo_produto", "id_track_global"]


def test_fieldmap_rename_dataframe(fmap):
    df = pd.DataFrame(
        {
            "id_tipo_produto": [1],
            "codigo_barra": ["X"],
            "extra_col": [99],  # not in map → stays
        }
    )
    renamed = fmap.rename_dataframe(df)
    assert "product_type_id" in renamed.columns
    assert "barcode" in renamed.columns
    assert "extra_col" in renamed.columns
    assert "id_tipo_produto" not in renamed.columns


def test_fieldmap_apply_column_names(fmap):
    df = pd.DataFrame({"product_type_id": [1], "barcode": ["X"]})
    result = fmap.apply_column_names(df, ["type_id", "code"])
    assert list(result.columns) == ["type_id", "code"]


def test_fieldmap_apply_column_names_wrong_count(fmap):
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="column_names length"):
        fmap.apply_column_names(df, ["only_one"])


class TestTranslateFiltersToDB:
    """semantic→DB translation used internally by _build_configured_request."""

    def test_simple_semantic(self, fmap):
        result = fmap.translate_filters_to_db({"product_type_id": 1}, input_keys_are="semantic")
        assert result == {"id_tipo_produto": 1}

    def test_with_op_suffix(self, fmap):
        result = fmap.translate_filters_to_db(
            {"product_type_id__gte": 1}, input_keys_are="semantic"
        )
        assert result == {"id_tipo_produto__gte": 1}

    def test_with_casting_and_op(self, fmap):
        result = fmap.translate_filters_to_db(
            {"global_track_id__date__gte": "2024-01-01"}, input_keys_are="semantic"
        )
        assert result == {"id_track_global__date__gte": "2024-01-01"}

    def test_db_mode_passthrough(self, fmap):
        """input_keys_are='db' → no translation (already DB names)."""
        original = {"id_tipo_produto__exact": 1}
        assert fmap.translate_filters_to_db(original, input_keys_are="db") == original

    def test_nested_or(self, fmap):
        filters = {
            "$or": [
                {"product_type_id": 1},
                {"global_track_id__gt": 100},
            ]
        }
        result = fmap.translate_filters_to_db(filters, input_keys_are="semantic")
        assert result == {
            "$or": [
                {"id_tipo_produto": 1},
                {"id_track_global__gt": 100},
            ]
        }

    def test_nested_and_not(self, fmap):
        filters = {
            "$and": [{"product_type_id__exact": 1}],
            "$not": {"barcode__isnull": True},
        }
        result = fmap.translate_filters_to_db(filters, input_keys_are="semantic")
        assert result == {
            "$and": [{"id_tipo_produto__exact": 1}],
            "$not": {"codigo_barra__isnull": True},
        }

    def test_unknown_key_passes_through(self, fmap):
        result = fmap.translate_filters_to_db(
            {"no_mapping_here": "value"}, input_keys_are="semantic"
        )
        assert result == {"no_mapping_here": "value"}

    def test_empty_field_map_passthrough(self):
        empty = FieldMap({})
        original = {"semantic_name__gt": 5}
        assert empty.translate_filters_to_db(original, input_keys_are="semantic") == original


# ===========================================================================
# DataGateway configured mode — SQLite in-memory databases
# ===========================================================================
#
# Two table structures are used:
#
#   LegacyProduct  — DB column names are the NON-semantic (legacy) names
#                    (id_tipo_produto, id_track_global, …)
#                    Used with field_map present.
#
#   SemanticProduct — DB column names ARE the semantic names
#                    (product_type_id, global_track_id, …)
#                    Used without a field_map.
# ---------------------------------------------------------------------------


class LegacyBase(DeclarativeBase):
    pass


class LegacyProduct(LegacyBase):
    """Table whose DB column names are the legacy (non-semantic) names."""

    __tablename__ = "legacy_products"

    id: Mapped[int] = mapped_column(primary_key=True)
    id_tipo_produto: Mapped[int] = mapped_column()
    id_track_global: Mapped[int] = mapped_column()
    id_track_proceso: Mapped[int] = mapped_column()
    codigo_barra: Mapped[str] = mapped_column(String(32))
    nombre_th: Mapped[str] = mapped_column(String(64))


class SemanticBase(DeclarativeBase):
    pass


class SemanticProduct(SemanticBase):
    """Table whose DB column names are already the semantic names."""

    __tablename__ = "semantic_products"

    id: Mapped[int] = mapped_column(primary_key=True)
    product_type_id: Mapped[int] = mapped_column()
    global_track_id: Mapped[int] = mapped_column()
    process_track_id: Mapped[int] = mapped_column()
    barcode: Mapped[str] = mapped_column(String(32))
    first_name: Mapped[str] = mapped_column(String(64))


ROWS = [
    # (type_id, global_track, process_track, barcode, name)
    (1, 10, 100, "A001", "Alice"),
    (1, 20, 200, "B002", "Bob"),
    (2, 30, 300, "C003", "Carol"),
]


@pytest.fixture(scope="module")
def legacy_dsn(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "legacy.db"
    engine = create_engine(f"sqlite:///{db_path}")
    LegacyBase.metadata.create_all(engine)
    with Session(engine) as s:
        s.add_all(
            [
                LegacyProduct(
                    id_tipo_produto=t,
                    id_track_global=g,
                    id_track_proceso=p,
                    codigo_barra=b,
                    nombre_th=n,
                )
                for t, g, p, b, n in ROWS
            ]
        )
        s.commit()
    engine.dispose()
    return f"sqlite:///{db_path}"


@pytest.fixture(scope="module")
def semantic_dsn(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "semantic.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SemanticBase.metadata.create_all(engine)
    with Session(engine) as s:
        s.add_all(
            [
                SemanticProduct(
                    product_type_id=t,
                    global_track_id=g,
                    process_track_id=p,
                    barcode=b,
                    first_name=n,
                )
                for t, g, p, b, n in ROWS
            ]
        )
        s.commit()
    engine.dispose()
    return f"sqlite:///{db_path}"


def _legacy_gw(dsn, **kwargs) -> DataGateway:
    """DataGateway over the legacy-column DB (field_map present → translate)."""
    config = SqlDatabaseConfig(connection_url=SecretStr(dsn), query_only=False)
    return DataGateway(
        config,
        table="legacy_products",
        field_map=PRODUCT_MAP,
        **kwargs,
    )


def _semantic_gw(dsn, **kwargs) -> DataGateway:
    """DataGateway over the semantic-column DB (no field_map → no translation)."""
    config = SqlDatabaseConfig(connection_url=SecretStr(dsn), query_only=False)
    return DataGateway(
        config,
        table="semantic_products",
        **kwargs,
    )


def _global_track_ids(frame):
    if isinstance(frame, dd.DataFrame):
        return frame.compute()["global_track_id"].tolist()
    if isinstance(frame, pd.DataFrame):
        return frame["global_track_id"].tolist()
    if isinstance(frame, pa.Table):
        return frame["global_track_id"].to_pylist()
    if isinstance(frame, pl.DataFrame):
        return frame["global_track_id"].to_list()
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


# ---------------------------------------------------------------------------
# field_map present  (DB has non-semantic column names → translate)
# ---------------------------------------------------------------------------


def test_legacy_sticky_filter_uses_semantic_names(legacy_dsn):
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


def test_legacy_runtime_filter_uses_semantic_names(legacy_dsn):
    """Runtime aload() kwargs are semantic names."""
    gw = _legacy_gw(legacy_dsn, sticky_filters={"product_type_id": 1})
    try:
        df = gw.load(global_track_id=10, as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["global_track_id"] == 10
    finally:
        gw.close()


def test_legacy_output_columns_are_semantic(legacy_dsn):
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


def test_legacy_gateway_arrow_output_uses_semantic_columns(legacy_dsn):
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
def test_legacy_gateway_return_type_matrix(legacy_dsn, return_type, expected_type):
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


def test_legacy_fieldnames_are_semantic(legacy_dsn):
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


def test_legacy_column_names_positional_override(legacy_dsn):
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


def test_legacy_or_filter_with_semantic_names(legacy_dsn):
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


def test_semantic_sticky_filter(semantic_dsn):
    """When no field_map is provided no translation or rename occurs."""
    gw = _semantic_gw(semantic_dsn, sticky_filters={"product_type_id": 1})
    try:
        df = gw.load(as_pandas=True)
        assert len(df) == 2
        assert "product_type_id" in df.columns
    finally:
        gw.close()


def test_semantic_runtime_filter(semantic_dsn):
    gw = _semantic_gw(semantic_dsn, sticky_filters={"product_type_id": 1})
    try:
        df = gw.load(global_track_id=10, as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["global_track_id"] == 10
    finally:
        gw.close()


def test_semantic_no_rename_applied(semantic_dsn):
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


def test_from_config_mirrors_dfhelper(legacy_dsn):
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


def test_from_config_preserves_embedded_fs_when_override_is_none(tmp_path):
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


def test_from_config_sqlalchemy_default_backend_error_names_the_fallback():
    """Omitting 'backend' silently defaults to sqlalchemy; the connection_url
    error should say so instead of looking like an unrelated validation bug."""
    with pytest.raises(ValueError, match="default used when no 'backend' key"):
        DataGateway.from_config({"parquet_storage_path": "/tmp/whatever"})


# ---------------------------------------------------------------------------
# DataFrameParams: chunk_size, index_col
# DataFrameOptions: sort_field, duplicate_expr/keep
# ---------------------------------------------------------------------------


def test_df_params_index_col(legacy_dsn):
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


def test_df_params_chunk_size_from_config(legacy_dsn):
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


def test_df_options_sort_field(legacy_dsn):
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


def test_df_options_dedup(legacy_dsn):
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


def test_df_options_from_config(legacy_dsn):
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


def test_exclude_negates_sticky_filter(legacy_dsn):
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


def test_exclude_with_runtime_filter(legacy_dsn):
    """exclude=True with a runtime kwarg excludes the matched row."""
    gw = _legacy_gw(legacy_dsn, exclude=True)
    try:
        df = gw.load(global_track_id=10, as_pandas=True)
        # Row with global_track_id=10 is excluded; 2 rows remain
        assert len(df) == 2
        assert 10 not in df["global_track_id"].tolist()
    finally:
        gw.close()


def test_use_exclude_compat_key(legacy_dsn):
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


def test_filter_to_fieldnames_sql_selects_subset(legacy_dsn):
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


def test_filter_to_fieldnames_missing_col_warns():
    """Requesting a fieldname absent from the loaded DataFrame emits a UserWarning."""
    import warnings

    from boti_data.gateway import DataFrameOptions
    from boti_data.gateway.post_process import PostProcessor

    pp = PostProcessor(
        FieldMap({}),
        DataFrameParams(fieldnames=("global_track_id", "nonexistent_col")),
        DataFrameOptions(),
    )
    df = pd.DataFrame({"global_track_id": [10, 20], "barcode": ["A", "B"]})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = pp.filter_to_fieldnames(df)

    assert any("nonexistent_col" in str(warning.message) for warning in w)
    assert "global_track_id" in result.columns
    assert "nonexistent_col" not in result.columns


# ===========================================================================
# Items 1–5: load_period, persist, timeout, _has_any_rows, chunked __in
# ===========================================================================


# ---------------------------------------------------------------------------
# Item 4: _has_any_rows
# ---------------------------------------------------------------------------


def test_has_any_rows_true_pandas():
    df = pd.DataFrame({"a": [1, 2]})
    assert DataGateway._has_any_rows(df) is True


def test_has_any_rows_false_pandas():
    df = pd.DataFrame({"a": []})
    assert DataGateway._has_any_rows(df) is False


def test_has_any_rows_no_columns():
    df = pd.DataFrame()
    assert DataGateway._has_any_rows(df) is False


# ---------------------------------------------------------------------------
# Item 2: persist=True
# ---------------------------------------------------------------------------


def test_load_persist_kwarg_accepted_and_returns_df(legacy_dsn):
    """persist=True is consumed as a control kwarg; result is a valid DataFrame."""
    gw = _legacy_gw(legacy_dsn)
    try:
        df = gw.load(persist=True, as_pandas=True)
        assert len(df) == 3
    finally:
        gw.close()


# ---------------------------------------------------------------------------
# Item 3: timeout=
# ---------------------------------------------------------------------------


def test_aload_timeout_not_exceeded(legacy_dsn):
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    try:
        df = asyncio.run(gw.aload(timeout=30, as_pandas=True))
        assert len(df) == 3
    finally:
        gw.close()


def test_aload_timeout_raises_when_exceeded(legacy_dsn, monkeypatch):
    """When timeout is hit asyncio.TimeoutError propagates."""
    import asyncio

    from boti_data.gateway.configured_load import ConfiguredLoadService

    original = ConfiguredLoadService.aload

    async def _slow(self, *args, **kwargs):
        await asyncio.sleep(10)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ConfiguredLoadService, "aload", _slow)
    gw = _legacy_gw(legacy_dsn)
    try:
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(gw.aload(timeout=0.01, as_pandas=True))
    finally:
        gw.close()


# ---------------------------------------------------------------------------
# Item 1: load_period / aload_period
# ---------------------------------------------------------------------------

# We test period filtering using SemanticProduct because the "date" column
# is not present in the existing test fixtures.  We create a lightweight new
# table with a date column inline.


import datetime as _dt


class DateBase(DeclarativeBase):
    pass


class EventRow(DateBase):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(32))
    event_date: Mapped[_dt.date] = mapped_column()


@pytest.fixture(scope="module")
def events_dsn(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "events.db"
    engine = create_engine(f"sqlite:///{db_path}")
    DateBase.metadata.create_all(engine)
    with Session(engine) as s:
        s.add_all(
            [
                EventRow(name="alpha", event_date=_dt.date(2024, 1, 10)),
                EventRow(name="beta", event_date=_dt.date(2024, 2, 15)),
                EventRow(name="gamma", event_date=_dt.date(2024, 3, 20)),
            ]
        )
        s.commit()
    engine.dispose()
    return f"sqlite:///{db_path}"


def _event_gw(dsn: str) -> DataGateway:
    config = SqlDatabaseConfig(connection_url=SecretStr(dsn), query_only=False)
    return DataGateway(config, table="events")


def test_load_period_single_date(events_dsn):
    """Single-date period (start == end) adds a ``field__date`` filter."""
    gw = _event_gw(events_dsn)
    try:
        df = gw.load_period("event_date", "2024-02-15", "2024-02-15", as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["name"] == "beta"
    finally:
        gw.close()


def test_load_period_range(events_dsn):
    """Date-range period adds a ``field__date__range`` filter."""
    gw = _event_gw(events_dsn)
    try:
        df = gw.load_period("event_date", "2024-01-01", "2024-02-28", as_pandas=True)
        assert len(df) == 2
        names = set(df["name"].tolist())
        assert names == {"alpha", "beta"}
    finally:
        gw.close()


def test_load_period_invalid_range_raises(events_dsn):
    gw = _event_gw(events_dsn)
    try:
        with pytest.raises(ValueError, match="'start' date cannot be later"):
            gw.load_period("event_date", "2024-12-31", "2024-01-01")
    finally:
        gw.close()


def test_aload_period_range(events_dsn):
    import asyncio

    gw = _event_gw(events_dsn)
    try:
        df = asyncio.run(gw.aload_period("event_date", "2024-01-01", "2024-03-31", as_pandas=True))
        assert len(df) == 3
    finally:
        gw.close()


# ---------------------------------------------------------------------------
# Item 5: chunked __in list splitting
# ---------------------------------------------------------------------------


def test_aload_chunked_in_splits_and_concatenates(legacy_dsn):
    """Oversized __in list is chunked; all rows are returned."""
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    # All 3 rows have global_track_id in [10, 20, 30]; chunk_size=1 forces 3 tasks
    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[10, 20, 30],
                in_chunk_size=1,
                as_pandas=True,
            )
        )
        assert len(df) == 3
        assert set(df["global_track_id"].tolist()) == {10, 20, 30}
    finally:
        gw.close()


def test_aload_chunked_in_no_split_when_within_limit(legacy_dsn):
    """List within chunk_size limit is sent as a single query."""
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[10, 20, 30],
                in_chunk_size=900,
                as_pandas=True,
            )
        )
        assert len(df) == 3
    finally:
        gw.close()


def test_aload_chunked_in_empty_result(legacy_dsn):
    """Chunking with values that match nothing returns an empty result."""
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[999, 888],
                in_chunk_size=1,
                as_pandas=True,
            )
        )
        assert len(df) == 0
    finally:
        gw.close()


def test_aload_chunked_in_supports_pandas_index_values(legacy_dsn):
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=pd.Index([10, 20, 30]),
                in_chunk_size=1,
                as_pandas=True,
            )
        )
        assert len(df) == 3
        assert set(df["global_track_id"].tolist()) == {10, 20, 30}
    finally:
        gw.close()


def test_aload_chunked_in_preserves_other_filters(legacy_dsn):
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[10, 20, 30],
                product_type_id=1,
                in_chunk_size=1,
                in_chunk_concurrency=2,
                as_pandas=True,
            )
        )
        assert set(df["global_track_id"].tolist()) == {10, 20}
        assert set(df["product_type_id"].tolist()) == {1}
    finally:
        gw.close()


def test_aload_chunked_in_honors_concurrency_cap():
    import asyncio

    gw = DataGateway(
        SqlDatabaseConfig(
            connection_url="sqlite:///:memory:",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
        )
    )

    current = 0
    max_seen = 0

    async def execute_fn(**opts):
        nonlocal current, max_seen
        current += 1
        max_seen = max(max_seen, current)
        await asyncio.sleep(0.01)
        current -= 1
        return pd.DataFrame({"value": list(opts["field__in"])})

    try:
        result = asyncio.run(
            gw._chunked_in_load(
                execute_fn,
                1,
                {"field__in": [1, 2, 3, 4]},
                return_type="pandas",
                max_concurrency=2,
            )
        )
        assert max_seen == 2
        assert result["value"].tolist() == [1, 2, 3, 4]
    finally:
        gw.close()


def test_aload_chunked_in_accepts_public_concurrency_option(legacy_dsn):
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[10, 20, 30],
                in_chunk_size=1,
                in_chunk_concurrency=2,
                as_pandas=True,
            )
        )
        assert len(df) == 3
        assert set(df["global_track_id"].tolist()) == {10, 20, 30}
    finally:
        gw.close()


def test_aload_chunked_in_auto_uses_filter_hints(legacy_dsn, monkeypatch):
    import asyncio

    gw = _legacy_gw(
        legacy_dsn,
        policies=GatewayPolicies(
            in_chunk_policy=InChunkPolicy(eager_auto_min_values=1, eager_auto_concurrency=2),
        ),
    )
    captured: dict[str, Any] = {}

    original = gw._chunked_in_load

    async def wrapped(execute_fn, chunk_size, options, *, return_type, max_concurrency=None):
        captured["chunk_size"] = chunk_size
        captured["max_concurrency"] = max_concurrency
        return await original(
            execute_fn,
            chunk_size,
            options,
            return_type=return_type,
            max_concurrency=max_concurrency,
        )

    monkeypatch.setattr(gw, "_chunked_in_load", wrapped)

    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[10, 20, 30],
                as_pandas=True,
            )
        )
        assert len(df) == 3
        assert captured == {"chunk_size": 1, "max_concurrency": 2}
    finally:
        gw.close()


def test_aload_chunked_in_off_disables_auto_hint(legacy_dsn, monkeypatch):
    import asyncio

    gw = _legacy_gw(legacy_dsn)
    captured: dict[str, Any] = {}

    original = gw._chunked_in_load

    async def wrapped(execute_fn, chunk_size, options, *, return_type, max_concurrency=None):
        captured["chunk_size"] = chunk_size
        captured["max_concurrency"] = max_concurrency
        return await original(
            execute_fn,
            chunk_size,
            options,
            return_type=return_type,
            max_concurrency=max_concurrency,
        )

    monkeypatch.setattr(gw, "_chunked_in_load", wrapped)

    try:
        df = asyncio.run(
            gw.aload(
                global_track_id__in=[10, 20, 30],
                in_chunk_strategy="off",
                as_pandas=True,
            )
        )
        assert len(df) == 3
        assert captured == {"chunk_size": 0, "max_concurrency": None}
    finally:
        gw.close()


def test_load_chunked_in_accepts_public_concurrency_option(legacy_dsn):
    gw = _legacy_gw(legacy_dsn)
    try:
        df = gw.load(
            global_track_id__in=[10, 20, 30],
            in_chunk_size=1,
            in_chunk_concurrency=2,
            as_pandas=True,
        )
        assert len(df) == 3
        assert set(df["global_track_id"].tolist()) == {10, 20, 30}
    finally:
        gw.close()


def test_load_chunked_in_auto_uses_filter_hints(legacy_dsn, monkeypatch):
    gw = _legacy_gw(
        legacy_dsn,
        policies=GatewayPolicies(
            in_chunk_policy=InChunkPolicy(eager_auto_min_values=1, eager_auto_concurrency=2),
        ),
    )
    captured: dict[str, Any] = {}

    original = gw._chunked_in_load_sync

    def wrapped(execute_fn, chunk_size, options, *, return_type, max_concurrency=None):
        captured["chunk_size"] = chunk_size
        captured["max_concurrency"] = max_concurrency
        return original(
            execute_fn,
            chunk_size,
            options,
            return_type=return_type,
            max_concurrency=max_concurrency,
        )

    monkeypatch.setattr(gw, "_chunked_in_load_sync", wrapped)

    try:
        df = gw.load(global_track_id__in=[10, 20, 30], as_pandas=True)
        assert len(df) == 3
        assert captured == {"chunk_size": 1, "max_concurrency": 2}
    finally:
        gw.close()


def test_load_chunked_in_explicit_values_override_auto_hint(legacy_dsn, monkeypatch):
    gw = _legacy_gw(legacy_dsn)
    captured: dict[str, Any] = {}

    original = gw._chunked_in_load_sync

    def wrapped(execute_fn, chunk_size, options, *, return_type, max_concurrency=None):
        captured["chunk_size"] = chunk_size
        captured["max_concurrency"] = max_concurrency
        return original(
            execute_fn,
            chunk_size,
            options,
            return_type=return_type,
            max_concurrency=max_concurrency,
        )

    monkeypatch.setattr(gw, "_chunked_in_load_sync", wrapped)

    try:
        df = gw.load(
            global_track_id__in=[10, 20, 30],
            in_chunk_size=2,
            in_chunk_concurrency=1,
            as_pandas=True,
        )
        assert len(df) == 3
        assert captured == {"chunk_size": 2, "max_concurrency": 1}
    finally:
        gw.close()


def test_aload_chunked_in_supports_nested_filter_payloads(legacy_dsn):
    import asyncio

    gw = DataGateway(SqlDatabaseConfig(connection_url=SecretStr(legacy_dsn), query_only=False))
    try:
        statement = select(LegacyProduct)
        df = asyncio.run(
            gw.aload(
                statement=statement,
                model=LegacyProduct,
                filters={"id_track_global__in": [10, 20, 30]},
                in_chunk_size=1,
                in_chunk_concurrency=2,
                as_pandas=True,
            )
        )
        assert set(df["id_track_global"].tolist()) == {10, 20, 30}
    finally:
        gw.close()


def test_load_chunked_in_preserves_other_filters(legacy_dsn):
    gw = _legacy_gw(legacy_dsn)
    try:
        df = gw.load(
            global_track_id__in=[10, 20, 30],
            product_type_id=1,
            in_chunk_size=1,
            in_chunk_concurrency=2,
            as_pandas=True,
        )
        assert set(df["global_track_id"].tolist()) == {10, 20}
        assert set(df["product_type_id"].tolist()) == {1}
    finally:
        gw.close()


def test_load_chunked_in_honors_concurrency_cap_sync():
    import threading
    import time

    gw = DataGateway(
        SqlDatabaseConfig(
            connection_url="sqlite:///:memory:",
            poolclass="sqlalchemy.pool.NullPool",
            query_only=False,
        )
    )

    current = 0
    max_seen = 0
    lock = threading.Lock()

    def execute_fn(**opts):
        nonlocal current, max_seen
        with lock:
            current += 1
            max_seen = max(max_seen, current)
        time.sleep(0.01)
        with lock:
            current -= 1
        return pd.DataFrame({"value": list(opts["field__in"])})

    try:
        result = gw._chunked_in_load_sync(
            execute_fn,
            1,
            {"field__in": [1, 2, 3, 4]},
            return_type="pandas",
            max_concurrency=2,
        )
        assert max_seen == 2
        assert result["value"].tolist() == [1, 2, 3, 4]
    finally:
        gw.close()


# ---------------------------------------------------------------------------
# Semi-join: Series as __in filter value (item 6)
# ---------------------------------------------------------------------------


def test_load_pandas_series_as_in_filter(legacy_dsn):
    """pd.Series passed as field__in resolves to unique-value list before query."""
    gw = _legacy_gw(legacy_dsn)
    try:
        series = pd.Series([10, 20, 10])  # duplicate 10 should be deduped
        df = gw.load(global_track_id__in=series, as_pandas=True)
        assert len(df) == 2
        assert set(df["global_track_id"].tolist()) == {10, 20}
    finally:
        gw.close()


def test_load_pandas_series_with_nan_drops_nulls(legacy_dsn):
    """NaN values in a pd.Series __in filter are silently discarded."""

    gw = _legacy_gw(legacy_dsn)
    try:
        series = pd.Series([10.0, float("nan")])
        df = gw.load(global_track_id__in=series, as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["global_track_id"] == 10
    finally:
        gw.close()


def test_load_dask_series_as_in_filter(legacy_dsn):
    """dd.Series passed as field__in is computed and resolved before query."""
    import dask.dataframe as dd

    gw = _legacy_gw(legacy_dsn)
    try:
        pdf = pd.DataFrame({"id": [10, 30, 30]})
        ddf = dd.from_pandas(pdf, npartitions=2)
        df = gw.load(global_track_id__in=ddf["id"], as_pandas=True)
        assert len(df) == 2
        assert set(df["global_track_id"].tolist()) == {10, 30}
    finally:
        gw.close()


@pytest.mark.asyncio
async def test_aload_pandas_series_as_in_filter(legacy_dsn):
    """Async path: pd.Series __in resolves correctly."""
    gw = _legacy_gw(legacy_dsn)
    try:
        series = pd.Series([20, 30])
        df = await gw.aload(global_track_id__in=series, as_pandas=True)
        assert len(df) == 2
        assert set(df["global_track_id"].tolist()) == {20, 30}
    finally:
        gw.close()


@pytest.mark.asyncio
async def test_aload_dask_series_as_in_filter(legacy_dsn):
    """Async path: dd.Series is computed in a thread pool before query."""
    import dask.dataframe as dd

    gw = _legacy_gw(legacy_dsn)
    try:
        pdf = pd.DataFrame({"gid": [10, 20]})
        ddf = dd.from_pandas(pdf, npartitions=1)
        df = await gw.aload(global_track_id__in=ddf["gid"], as_pandas=True)
        assert len(df) == 2
        assert set(df["global_track_id"].tolist()) == {10, 20}
    finally:
        gw.close()


def test_semi_join_method(legacy_dsn):
    """semi_join() is sugar for load(field__in=series)."""
    gw = _legacy_gw(legacy_dsn)
    try:
        series = pd.Series([10, 30])
        df = gw.semi_join(series, on="global_track_id", as_pandas=True)
        assert len(df) == 2
        assert set(df["global_track_id"].tolist()) == {10, 30}
    finally:
        gw.close()


@pytest.mark.asyncio
async def test_asemi_join_method(legacy_dsn):
    """asemi_join() is sugar for aload(field__in=series)."""
    gw = _legacy_gw(legacy_dsn)
    try:
        series = pd.Series([20])
        df = await gw.asemi_join(series, on="global_track_id", as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["global_track_id"] == 20
    finally:
        gw.close()


def test_semi_join_with_field_map_translates_column(legacy_dsn):
    """semi_join uses semantic names; field_map handles DB translation."""
    gw = _legacy_gw(legacy_dsn, sticky_filters={"product_type_id": 1})
    try:
        # Both sticky filter (type=1) AND semi-join must apply
        series = pd.Series([10])
        df = gw.semi_join(series, on="global_track_id", as_pandas=True)
        assert len(df) == 1
        assert df.iloc[0]["product_type_id"] == 1
        assert df.iloc[0]["global_track_id"] == 10
        assert "id_track_global" not in df.columns
    finally:
        gw.close()


def test_resolve_series_filters_is_noop_without_series():
    """resolve_series_filters returns the same dict when no Series are present."""
    opts = {"field__in": [1, 2, 3], "limit": 10}
    result = gateway_series_filters.resolve_series_filters(opts)
    assert result is opts  # same object, no copy made


# ---------------------------------------------------------------------------
# Laziness guarantees — load() must return dd.DataFrame unless as_pandas=True
# ---------------------------------------------------------------------------


def test_load_returns_lazy_dask_by_default(legacy_dsn):
    """load() without as_pandas returns a dd.DataFrame, not a pandas DataFrame."""
    gw = _legacy_gw(legacy_dsn)
    try:
        result = gw.load()
        assert isinstance(result, dd.DataFrame), (
            f"Expected dd.DataFrame, got {type(result).__name__}"
        )
    finally:
        gw.close()


def test_load_as_pandas_returns_pandas(legacy_dsn):
    """load(as_pandas=True) returns a pandas DataFrame."""
    gw = _legacy_gw(legacy_dsn)
    try:
        result = gw.load(as_pandas=True)
        assert isinstance(result, pd.DataFrame), (
            f"Expected pd.DataFrame, got {type(result).__name__}"
        )
    finally:
        gw.close()


def test_configured_gateway_auto_return_type_prefers_pandas_for_small_sql(legacy_dsn):
    gw = _legacy_gw(legacy_dsn, df_params=DataFrameParams(return_type="auto"))
    try:
        result = gw.load()
        assert isinstance(result, pd.DataFrame)
    finally:
        gw.close()


def test_configured_gateway_auto_uses_eager_fetch_for_small_sql(legacy_dsn, monkeypatch):
    eager_calls: list[bool] = []
    lazy_calls: list[bool] = []
    real_load_sql = load_sql
    real_load_sql_partitioned = load_sql_partitioned

    def tracking_load_sql(resource, request):
        eager_calls.append(True)
        return real_load_sql(resource, request)

    def tracking_load_sql_partitioned(config, resource, request):
        lazy_calls.append(True)
        return real_load_sql_partitioned(config, resource, request)

    monkeypatch.setattr("boti_data.gateway._backend_strategies.load_sql", tracking_load_sql)
    monkeypatch.setattr(
        "boti_data.gateway._backend_strategies.load_sql_partitioned", tracking_load_sql_partitioned
    )
    gw = _legacy_gw(legacy_dsn, df_params=DataFrameParams(return_type="auto"))
    try:
        result = gw.load()
        assert isinstance(result, pd.DataFrame)
        assert eager_calls == [True]
        assert not lazy_calls
    finally:
        gw.close()


def test_configured_gateway_reflects_table_once(legacy_dsn, monkeypatch):
    calls: list[bool] = []
    real_reflect_and_select = gateway_core.reflect_and_select

    def tracking_reflect_and_select(*args, **kwargs):
        calls.append(True)
        return real_reflect_and_select(*args, **kwargs)

    monkeypatch.setattr(gateway_core, "reflect_and_select", tracking_reflect_and_select)
    gw = _legacy_gw(legacy_dsn)
    try:
        first = gw.load(as_pandas=True)
        second = gw.load(as_pandas=True)
        assert isinstance(first, pd.DataFrame)
        assert isinstance(second, pd.DataFrame)
        assert calls == [True]
    finally:
        gw.close()


def test_configured_gateway_eager_sql_bypasses_internal_request_revalidation(
    legacy_dsn, monkeypatch
):
    gw = _legacy_gw(
        legacy_dsn,
        df_params=DataFrameParams(return_type="pandas", execution_mode="eager"),
    )
    monkeypatch.setattr(
        SqlLoadRequest,
        "model_validate",
        classmethod(
            lambda cls, *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("model_validate should not run in configured eager SQL path")
            )
        ),
    )
    try:
        result = gw.load()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    finally:
        gw.close()


def test_configured_gateway_auto_return_type_uses_dask_for_large_sql(legacy_dsn, monkeypatch):
    monkeypatch.setattr(gateway_return_type, "_AUTO_EAGER_MAX_ROWS", 1)
    gw = _legacy_gw(legacy_dsn, df_params=DataFrameParams(return_type="auto"))
    try:
        result = gw.load()
        assert isinstance(result, dd.DataFrame)
    finally:
        gw.close()


@pytest.mark.asyncio
async def test_async_series_filter_resolution_batches_dask_compute(monkeypatch):
    left = dd.from_pandas(pd.DataFrame({"id": [1, 2, 3]}), npartitions=2)["id"]
    right = dd.from_pandas(pd.DataFrame({"id": [10, 20, 30]}), npartitions=2)["id"]
    calls: list[int] = []
    real_compute = gateway_series_filters.dask.compute

    def tracking_compute(*args, **kwargs):
        calls.append(len(args))
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(gateway_series_filters.dask, "compute", tracking_compute)

    resolved = await gateway_series_filters.resolve_series_filters_async(
        {"left__in": left, "right__in": right}
    )

    assert resolved["left__in"] == [1, 2, 3]
    assert resolved["right__in"] == [10, 20, 30]
    assert calls == [2]


def test_lazy_result_has_correct_columns(legacy_dsn):
    """Lazy dd.DataFrame must carry semantic column names (field_map applied)."""
    gw = _legacy_gw(legacy_dsn)
    try:
        result = gw.load()
        assert isinstance(result, dd.DataFrame)
        assert "product_type_id" in result.columns
        assert "id_tipo_produto" not in result.columns
    finally:
        gw.close()


def test_lazy_result_fieldnames_pipeline(legacy_dsn):
    """fieldnames filter preserves laziness — result is still a dd.DataFrame."""
    gw = _legacy_gw(
        legacy_dsn,
        df_params=DataFrameParams(fieldnames=("global_track_id", "barcode")),
    )
    try:
        result = gw.load()
        assert isinstance(result, dd.DataFrame)
        assert set(result.columns) == {"global_track_id", "barcode"}
    finally:
        gw.close()


def test_lazy_result_sticky_filter_preserves_dask(legacy_dsn):
    """Sticky filters must not force an eager compute."""
    gw = _legacy_gw(legacy_dsn, sticky_filters={"product_type_id": 1})
    try:
        result = gw.load()
        assert isinstance(result, dd.DataFrame)
        # Compute only here — result must materialise correctly
        pdf = result.compute()
        assert len(pdf) == 2
        assert set(pdf["product_type_id"].tolist()) == {1}
    finally:
        gw.close()


def test_lazy_result_computes_correct_data(legacy_dsn):
    """Lazy result must compute to the same data as as_pandas=True."""
    gw = _legacy_gw(legacy_dsn)
    try:
        lazy = gw.load()
        assert isinstance(lazy, dd.DataFrame)
        eager = gw.load(as_pandas=True)
        assert isinstance(eager, pd.DataFrame)
        pd.testing.assert_frame_equal(
            lazy.compute().sort_values("global_track_id").reset_index(drop=True),
            eager.sort_values("global_track_id").reset_index(drop=True),
        )
    finally:
        gw.close()


@pytest.mark.asyncio
async def test_aload_returns_lazy_dask_by_default(legacy_dsn):
    """aload() without as_pandas returns a dd.DataFrame."""
    gw = _legacy_gw(legacy_dsn)
    try:
        result = await gw.aload()
        assert isinstance(result, dd.DataFrame), (
            f"Expected dd.DataFrame, got {type(result).__name__}"
        )
    finally:
        gw.close()


@pytest.mark.asyncio
async def test_aload_as_pandas_returns_pandas(legacy_dsn):
    """aload(as_pandas=True) returns a pandas DataFrame."""
    gw = _legacy_gw(legacy_dsn)
    try:
        result = await gw.aload(as_pandas=True)
        assert isinstance(result, pd.DataFrame)
    finally:
        gw.close()


def test_semi_join_without_as_pandas_returns_dask(legacy_dsn):
    """semi_join() default result is a lazy dd.DataFrame."""
    gw = _legacy_gw(legacy_dsn)
    try:
        result = gw.semi_join(pd.Series([10, 30]), on="global_track_id")
        assert isinstance(result, dd.DataFrame)
        pdf = result.compute()
        assert set(pdf["global_track_id"].tolist()) == {10, 30}
    finally:
        gw.close()


def test_lazy_semi_join_avoids_series_list_resolution(legacy_dsn, monkeypatch):
    gw = _legacy_gw(legacy_dsn)

    def forbid_series_resolution(options):
        if any(isinstance(value, (pd.Series, dd.Series)) for value in options.values()):
            raise AssertionError(
                "series should not reach generic filter resolution in lazy semi_join"
            )
        return options

    monkeypatch.setattr(gateway_series_filters, "resolve_series_filters", forbid_series_resolution)
    try:
        result = gw.semi_join(pd.Series([10, 30]), on="global_track_id")
        assert isinstance(result, dd.DataFrame)
        assert set(result.compute()["global_track_id"].tolist()) == {10, 30}
    finally:
        gw.close()
