"""
SQLAlchemy ↔ PyArrow type mapping and Arrow-based schema coercion.

Provides zero-copy type conversion from SQLAlchemy result rows to PyArrow
Tables, replacing the per-column pandas coercion loop with a single-pass
``table.cast()`` operation.
"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

import pyarrow as pa

try:
    import boti_rs as _boti_rs
    _HAS_RUST = True
except ImportError:
    _boti_rs = None  # type: ignore[assignment]
    _HAS_RUST = False
from sqlalchemy.sql.sqltypes import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    Numeric,
    SmallInteger,
    String,
    Text,
    Time,
    Unicode,
    UnicodeText,
)

# ---------------------------------------------------------------------------
# SQLAlchemy → PyArrow type mapping
# ---------------------------------------------------------------------------

def sqlalchemy_type_to_arrow(sql_type: Any) -> pa.DataType:
    """Map a SQLAlchemy type instance to a PyArrow DataType."""
    if isinstance(sql_type, (SmallInteger,)):
        return pa.int16()
    if isinstance(sql_type, (Integer,)):
        return pa.int32()
    if isinstance(sql_type, (BigInteger,)):
        return pa.int64()
    if isinstance(sql_type, (Numeric,)):
        precision = getattr(sql_type, "precision", None) or 18
        scale = getattr(sql_type, "scale", None) or 6
        return pa.decimal128(precision, scale)
    if isinstance(sql_type, (Float,)):
        return pa.float64()
    if isinstance(sql_type, (Boolean,)):
        return pa.bool_()
    if isinstance(sql_type, (Date,)):
        return pa.date32()
    if isinstance(sql_type, (DateTime,)):
        return pa.timestamp("ns", tz="UTC")
    if isinstance(sql_type, (Time,)):
        return pa.time64("ns")
    if isinstance(sql_type, (String, Text, Unicode, UnicodeText)):
        return pa.string()
    if isinstance(sql_type, (LargeBinary,)):
        return pa.binary()
    return pa.string()  # safe fallback


# ---------------------------------------------------------------------------
# Canonical pandas dtype → PyArrow type (for meta_dtypes compatibility)
# ---------------------------------------------------------------------------

_PANDAS_DTYPE_TO_ARROW: dict[str, pa.DataType] = {
    "Int64": pa.int64(),
    "Int32": pa.int32(),
    "Float64": pa.float64(),
    "boolean": pa.bool_(),
    "string": pa.string(),
    "datetime64[ns, UTC]": pa.timestamp("ns", tz="UTC"),
    "datetime64[ns]": pa.timestamp("ns"),
    "object": pa.string(),
}


def pandas_dtype_to_arrow(dtype_str: str) -> pa.DataType:
    """Convert a canonical pandas dtype string to a PyArrow DataType."""
    normalized = dtype_str.strip().lower()
    # Direct match
    for key, arrow_type in _PANDAS_DTYPE_TO_ARROW.items():
        if normalized == key.lower():
            return arrow_type
    # Fallback: try to parse common patterns
    if "int64" in normalized:
        return pa.int64()
    if "int32" in normalized:
        return pa.int32()
    if "float64" in normalized:
        return pa.float64()
    if "bool" in normalized:
        return pa.bool_()
    if "datetime" in normalized:
        if "utc" in normalized:
            return pa.timestamp("ns", tz="UTC")
        return pa.timestamp("ns")
    return pa.string()


# ---------------------------------------------------------------------------
# Schema builders
# ---------------------------------------------------------------------------

def build_arrow_schema_from_meta_dtypes(
    meta_dtypes: dict[str, str],
) -> pa.Schema:
    """Build a PyArrow Schema from a pandas-style meta_dtypes dict."""
    if _HAS_RUST:
        return _boti_rs.build_arrow_schema_from_meta_dtypes(meta_dtypes)
    fields = [
        (col, pandas_dtype_to_arrow(dtype))
        for col, dtype in meta_dtypes.items()
    ]
    return pa.schema(fields)


def build_arrow_schema_from_sqlalchemy_types(
    columns: list[str],
    sql_types: list[Any],
) -> pa.Schema:
    """Build a PyArrow Schema from SQLAlchemy column names and types."""
    fields = [
        (col, sqlalchemy_type_to_arrow(sql_type))
        for col, sql_type in zip(columns, sql_types)
    ]
    return pa.schema(fields)


# ---------------------------------------------------------------------------
# Value coercion helpers
# ---------------------------------------------------------------------------

def _coerce_value(value: Any, arrow_type: pa.DataType) -> Any:
    """Coerce a single Python value to be compatible with the Arrow type."""
    if value is None:
        return None
    if pa.types.is_timestamp(arrow_type):
        import pandas as pd

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None

        if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
            value = dt.datetime.combine(value, dt.time.min)

        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None

        if arrow_type.tz:
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(arrow_type.tz)
            else:
                timestamp = timestamp.tz_convert(arrow_type.tz)
        elif timestamp.tzinfo is not None:
            timestamp = timestamp.tz_localize(None)

        return timestamp.to_pydatetime()
    if pa.types.is_date(arrow_type):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            return dt.date.fromisoformat(value)
        if isinstance(value, dt.datetime):
            return value.date()
        return value
    if pa.types.is_integer(arrow_type):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(
                    f"Cannot coerce fractional float {value!r} to integer type "
                    f"{arrow_type} without data loss"
                )
            return int(value)
        if isinstance(value, Decimal):
            if value != value.to_integral_value():
                raise ValueError(
                    f"Cannot coerce fractional Decimal {value!r} to integer type "
                    f"{arrow_type} without data loss"
                )
            return int(value)
        return value
    if pa.types.is_floating(arrow_type):
        if isinstance(value, (int, float, Decimal)):
            return float(value)
        return value
    if pa.types.is_boolean(arrow_type):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "t"}
        return bool(value) if value is not None else None
    if pa.types.is_string(arrow_type):
        if isinstance(value, (dt.datetime, dt.date)):
            return value.isoformat()
        return str(value) if value is not None else None
    return value


def _coerce_row_to_arrays(
    rows: list[tuple],
    columns: list[str],
    schema: pa.Schema,
) -> list[pa.Array]:
    """Convert row tuples into columnar PyArrow Arrays with type coercion."""
    if not rows:
        return [pa.array([], type=field.type) for field in schema]

    arrays: list[pa.Array] = []
    for col_idx, field in enumerate(schema):
        raw_values = [row[col_idx] for row in rows]
        # Fast path: try direct array construction
        try:
            arr = pa.array(raw_values, type=field.type)
            arrays.append(arr)
            continue
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError):
            pass

        # Slow path: coerce individual values
        coerced = [_coerce_value(v, field.type) for v in raw_values]
        try:
            arr = pa.array(coerced, type=field.type)
            arrays.append(arr)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            # Last resort: use string type
            arr = pa.array([str(v) if v is not None else None for v in raw_values], type=pa.string())
            arrays.append(arr)

    return arrays


# ---------------------------------------------------------------------------
# Table construction from SQLAlchemy cursor results
# ---------------------------------------------------------------------------

def rows_to_arrow_table(
    rows: list[tuple],
    columns: list[str],
    schema: pa.Schema,
) -> pa.Table:
    """Build a PyArrow Table from SQLAlchemy cursor rows and an Arrow Schema.

    This replaces the slow ``pd.DataFrame(rows, columns=columns)`` path with
    a columnar Arrow construction that enables zero-copy ``table.cast()`` for
    schema coercion.
    Uses the Rust-accelerated implementation when available.
    """
    if _HAS_RUST:
        return _boti_rs.rows_to_arrow_table(rows, columns, schema)
    arrays = _coerce_row_to_arrays(rows, columns, schema)
    return pa.Table.from_arrays(arrays, schema=schema)


def build_empty_arrow_table(schema: pa.Schema) -> pa.Table:
    """Build an empty Arrow Table with the given schema (zero rows)."""
    if _HAS_RUST:
        return _boti_rs.build_empty_arrow_table(schema)
    arrays = [pa.array([], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


# ---------------------------------------------------------------------------
# Arrow → pandas conversion with type preservation
# ---------------------------------------------------------------------------

def arrow_table_to_pandas(
    table: pa.Table,
    *,
    types_mapper: Any | None = None,
) -> Any:
    """Convert an Arrow Table to pandas with proper type mapping.

    Uses pandas nullable dtypes (Int64, boolean, string) for Arrow types
    to maintain compatibility with existing boti schema contracts.
    """
    import pandas as pd

    def _default_types_mapper(arrow_dtype: pa.DataType) -> Any:
        if pa.types.is_int64(arrow_dtype):
            return pd.Int64Dtype()
        if pa.types.is_int32(arrow_dtype):
            return pd.Int32Dtype()
        if pa.types.is_float64(arrow_dtype):
            return pd.Float64Dtype()
        if pa.types.is_boolean(arrow_dtype):
            return pd.BooleanDtype()
        if pa.types.is_string(arrow_dtype) or pa.types.is_large_string(arrow_dtype):
            return pd.StringDtype()
        return None

    mapper = types_mapper if types_mapper is not None else _default_types_mapper
    dataframe = table.to_pandas(types_mapper=mapper)

    for field in table.schema:
        if field.name not in dataframe.columns:
            continue
        if pa.types.is_timestamp(field.type):
            if field.type.tz:
                dataframe[field.name] = pd.Series(
                    pd.to_datetime(dataframe[field.name], utc=True),
                    dtype=pd.DatetimeTZDtype(unit="ns", tz=field.type.tz),
                )
            else:
                dataframe[field.name] = pd.to_datetime(dataframe[field.name], errors="coerce")

    return dataframe


# ---------------------------------------------------------------------------
# Schema coercion: single-pass cast (replaces apply_schema_map loop)
# ---------------------------------------------------------------------------

def coerce_arrow_table(
    table: pa.Table,
    target_schema: pa.Schema,
) -> pa.Table:
    """Coerce an Arrow Table to a target schema in a single pass.

    Uses ``table.cast()`` for batch coercion instead of per-column pandas
    operations. Handles type mismatches gracefully via safe casting.
    """
    if table.schema == target_schema:
        return table

    arrays: list[pa.ChunkedArray] = []
    for target_field in target_schema:
        if target_field.name not in table.column_names:
            raise ValueError(
                f"Column '{target_field.name}' not found in source table. "
                f"Available: {table.schema.names}"
            )

        source_col = table.column(target_field.name)
        if source_col.type == target_field.type:
            arrays.append(source_col)
            continue

        try:
            arrays.append(source_col.cast(target_field.type, safe=True))
        except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
            raise pa.ArrowInvalid(
                f"Cannot safely cast column '{target_field.name}' from "
                f"{source_col.type!r} to {target_field.type!r}: {exc}"
            ) from exc

    return pa.Table.from_arrays(arrays, schema=target_schema)
