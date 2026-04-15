"""
Tests for shared DataFrame schema normalization utilities.
"""

from __future__ import annotations

import datetime as dt

import dask.dataframe as dd
import pandas as pd
import pytest

from boti_data.arrow_schema import ArrowSchema
from boti_data.db.arrow_schema_mapper import (
    arrow_table_to_pandas,
    build_arrow_schema_from_meta_dtypes,
    rows_to_arrow_table,
)
from boti_data.schema import (
    SchemaValidationError,
    align_frames_for_join,
    apply_schema_map,
    infer_schema_map,
    normalize_dtype_alias,
    normalize_schema_map,
    validate_schema,
)


def test_normalize_dtype_alias_handles_pyarrow_and_extension_variants():
    assert normalize_dtype_alias("int64[pyarrow]") == "Int64"
    assert normalize_dtype_alias("string[pyarrow]") == "string"
    assert normalize_dtype_alias("boolean[pyarrow]") == "boolean"
    assert normalize_dtype_alias("timestamp[ns, tz=UTC][pyarrow]") == "datetime64[ns, UTC]"
    assert normalize_dtype_alias("timestamp[s, tz=UTC][pyarrow]") == "datetime64[ns, UTC]"
    assert normalize_dtype_alias("datetime64[ns, utc]") == "datetime64[ns, UTC]"
    assert normalize_dtype_alias("datetime64[s, utc]") == "datetime64[ns, UTC]"
    assert normalize_dtype_alias("datetime64[s]") == "datetime64[ns]"


def test_apply_schema_map_and_validate_schema_for_pandas():
    dataframe = pd.DataFrame(
        {
            "id": ["1", "2"],
            "flag": ["yes", "0"],
            "ts": ["2024-01-01T00:00:00Z", "2024-01-02T12:30:00Z"],
        }
    )
    schema_map = {
        "id": "int64[pyarrow]",
        "flag": "boolean[pyarrow]",
        "ts": "datetime64[ns, UTC]",
    }

    aligned = apply_schema_map(dataframe, schema_map, require_columns=True)
    validate_schema(aligned, schema_map, require_columns=True)

    assert infer_schema_map(aligned) == {
        "id": "Int64",
        "flag": "boolean",
        "ts": "datetime64[ns, UTC]",
    }
    assert aligned["id"].tolist() == [1, 2]
    assert aligned["flag"].tolist() == [True, False]


def test_validate_schema_accepts_second_resolution_utc_datetimes():
    dataframe = pd.DataFrame(
        {
            "ts": pd.Series(
                pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-02T12:30:00Z"], utc=True)
            ).astype("datetime64[s, UTC]")
        }
    )

    validate_schema(dataframe, {"ts": "datetime64[ns, UTC]"}, require_columns=True)


def test_validate_schema_raises_on_dtype_mismatch():
    dataframe = pd.DataFrame({"id": [1, 2]})

    with pytest.raises(SchemaValidationError, match="expected 'string'"):
        validate_schema(dataframe, {"id": "string"})


def test_apply_schema_map_raises_on_missing_required_columns():
    dataframe = pd.DataFrame({"id": [1, 2]})

    with pytest.raises(SchemaValidationError, match="Missing required column"):
        apply_schema_map(dataframe, {"missing": "Int64"}, require_columns=True)


def test_align_frames_for_join_normalizes_dask_join_keys():
    left_pdf = pd.DataFrame(
        {
            "id": pd.Series([1, 2], dtype="Int64"),
            "left_value": ["a", "b"],
        }
    )
    right_pdf = pd.DataFrame(
        {
            "id": pd.Series(["1", "2"], dtype="string"),
            "right_value": ["x", "y"],
        }
    )

    left = dd.from_pandas(left_pdf, npartitions=2)
    right = dd.from_pandas(right_pdf, npartitions=1)

    left_aligned, right_aligned = align_frames_for_join(
        left,
        right,
        {"id": "Int64"},
    )

    validate_schema(left_aligned, {"id": "Int64"})
    validate_schema(right_aligned, {"id": "Int64"})

    merged = left_aligned.merge(right_aligned, on="id").compute().sort_values("id")

    assert normalize_schema_map(infer_schema_map(left_aligned, columns=["id"])) == {"id": "Int64"}
    assert merged["right_value"].tolist() == ["x", "y"]


def test_arrow_schema_roundtrips_empty_utc_timestamp_dataframe():
    dataframe = pd.DataFrame(
        {
            "id": pd.Series(dtype="Int64"),
            "ts": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )

    schema = ArrowSchema.from_dict({"id": "Int64", "ts": "datetime64[ns, UTC]"})
    result = schema.to_pandas(schema.cast_table(schema.to_arrow_table(dataframe)))

    assert str(result["id"].dtype) == "Int64"
    assert str(result["ts"].dtype) == "datetime64[ns, UTC]"


def test_rows_to_arrow_table_coerces_date_values_to_utc_timestamps():
    schema = build_arrow_schema_from_meta_dtypes({"ts": "datetime64[ns, UTC]"})

    table = rows_to_arrow_table(
        [(dt.date(2046, 5, 31),), ("2046-06-01",)],
        ["ts"],
        schema,
    )
    result = arrow_table_to_pandas(table)

    assert str(result["ts"].dtype) == "datetime64[ns, UTC]"
    assert result["ts"].tolist() == [
        pd.Timestamp("2046-05-31T00:00:00Z"),
        pd.Timestamp("2046-06-01T00:00:00Z"),
    ]


def test_rows_to_arrow_table_treats_blank_timestamp_strings_as_null():
    schema = build_arrow_schema_from_meta_dtypes({"ts": "datetime64[ns, UTC]"})

    table = rows_to_arrow_table(
        [("",), (None,), ("2046-05-31",)],
        ["ts"],
        schema,
    )
    result = arrow_table_to_pandas(table)

    assert str(result["ts"].dtype) == "datetime64[ns, UTC]"
    assert pd.isna(result.loc[0, "ts"])
    assert pd.isna(result.loc[1, "ts"])
    assert result.loc[2, "ts"] == pd.Timestamp("2046-05-31T00:00:00Z")
