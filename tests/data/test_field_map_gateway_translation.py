"""
FieldMap unit tests: to_db/to_semantic translation, renaming, and the
semantic→DB filter-key translation used internally by _build_configured_request.

Split out of test_field_map_gateway.py purely for god-module/long-file headroom.
"""

from __future__ import annotations

import pandas as pd
import pytest

from boti_data.field_map import FieldMap

from .conftest import PRODUCT_MAP


@pytest.fixture
def fmap() -> FieldMap:
    return FieldMap(PRODUCT_MAP)


# ===========================================================================
# FieldMap unit tests
# ===========================================================================


def test_fieldmap_to_db(fmap) -> None:
    assert fmap.to_db("product_type_id") == "id_tipo_produto"
    assert fmap.to_db("global_track_id") == "id_track_global"
    # Unknown semantic name passes through unchanged
    assert fmap.to_db("unknown_col") == "unknown_col"


def test_fieldmap_to_semantic(fmap) -> None:
    assert fmap.to_semantic("id_tipo_produto") == "product_type_id"
    # Unknown DB column passes through unchanged
    assert fmap.to_semantic("some_other_col") == "some_other_col"


def test_fieldmap_bool() -> None:
    assert bool(FieldMap(PRODUCT_MAP)) is True
    assert bool(FieldMap({})) is False


def test_fieldmap_duplicate_semantic_raises() -> None:
    with pytest.raises(ValueError, match="Duplicate semantic name"):
        FieldMap({"col_a": "same", "col_b": "same"})


def test_fieldmap_select_db_columns(fmap) -> None:
    result = fmap.select_db_columns(["product_type_id", "global_track_id"])
    assert result == ["id_tipo_produto", "id_track_global"]


def test_fieldmap_rename_dataframe(fmap) -> None:
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


def test_fieldmap_apply_column_names(fmap) -> None:
    df = pd.DataFrame({"product_type_id": [1], "barcode": ["X"]})
    result = fmap.apply_column_names(df, ["type_id", "code"])
    assert list(result.columns) == ["type_id", "code"]


def test_fieldmap_apply_column_names_wrong_count(fmap) -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="column_names length"):
        fmap.apply_column_names(df, ["only_one"])


class TestTranslateFiltersToDB:
    """semantic→DB translation used internally by _build_configured_request."""

    def test_simple_semantic(self, fmap) -> None:
        result = fmap.translate_filters_to_db({"product_type_id": 1}, input_keys_are="semantic")
        assert result == {"id_tipo_produto": 1}

    def test_with_op_suffix(self, fmap) -> None:
        result = fmap.translate_filters_to_db(
            {"product_type_id__gte": 1}, input_keys_are="semantic"
        )
        assert result == {"id_tipo_produto__gte": 1}

    def test_with_casting_and_op(self, fmap) -> None:
        result = fmap.translate_filters_to_db(
            {"global_track_id__date__gte": "2024-01-01"}, input_keys_are="semantic"
        )
        assert result == {"id_track_global__date__gte": "2024-01-01"}

    def test_db_mode_passthrough(self, fmap) -> None:
        """input_keys_are='db' → no translation (already DB names)."""
        original = {"id_tipo_produto__exact": 1}
        assert fmap.translate_filters_to_db(original, input_keys_are="db") == original

    def test_nested_or(self, fmap) -> None:
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

    def test_nested_and_not(self, fmap) -> None:
        filters = {
            "$and": [{"product_type_id__exact": 1}],
            "$not": {"barcode__isnull": True},
        }
        result = fmap.translate_filters_to_db(filters, input_keys_are="semantic")
        assert result == {
            "$and": [{"id_tipo_produto__exact": 1}],
            "$not": {"codigo_barra__isnull": True},
        }

    def test_unknown_key_passes_through(self, fmap) -> None:
        result = fmap.translate_filters_to_db(
            {"no_mapping_here": "value"}, input_keys_are="semantic"
        )
        assert result == {"no_mapping_here": "value"}

    def test_empty_field_map_passthrough(self) -> None:
        empty = FieldMap({})
        original = {"semantic_name__gt": 5}
        assert empty.translate_filters_to_db(original, input_keys_are="semantic") == original
