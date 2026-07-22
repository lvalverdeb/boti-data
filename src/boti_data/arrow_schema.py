"""
Arrow Schema Contract — PyArrow-backed schema definition and coercion.

Provides a canonical ``ArrowSchema`` class that wraps ``pa.Schema`` with:
- Single-pass ``cast_table()`` coercion (replaces per-column pandas loops)
- Built-in schema equality comparison
- Dict-compatible constructor for migration from existing schema maps
- DataFrame ↔ Table conversion helpers
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "ArrowSchema",
    "apply_arrow_schema_map",
    "validate_arrow_schema",
]

import pandas as pd
import pyarrow as pa


class ArrowSchema:
    """Canonical PyArrow schema contract with single-pass coercion.

    Usage::

        schema = ArrowSchema.from_dict({
            "col_a": "Int64",
            "col_b": "boolean",
            "col_c": "datetime64[ns, UTC]",
        })

        # Coerce any DataFrame to this schema in one pass
        table = schema.to_arrow_table(df)
        coerced_table = schema.cast_table(table)
        result_df = schema.to_dataframe(coerced_table)
    """

    def __init__(self, schema: pa.Schema) -> None:
        self._schema = schema

    @property
    def pa_schema(self) -> pa.Schema:
        """Return the underlying PyArrow Schema."""
        return self._schema

    @property
    def column_names(self) -> list[str]:
        return list(self._schema.names)

    @property
    def column_types(self) -> dict[str, pa.DataType]:
        return {field.name: field.type for field in self._schema}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, dtype_map: Mapping[str, str]) -> ArrowSchema:
        """Build from a pandas-style dtype mapping (same format as existing schema_map)."""
        from boti_data.db.arrow_schema_mapper import pandas_dtype_to_arrow
        from boti_data.schema import normalize_schema_map

        normalized = normalize_schema_map(dtype_map)
        fields = [(col, pandas_dtype_to_arrow(dtype)) for col, dtype in normalized.items()]
        return cls(pa.schema(fields))

    @classmethod
    def from_fields(cls, fields: list[tuple[str, pa.DataType]]) -> ArrowSchema:
        """Build from explicit (name, arrow_type) pairs."""
        return cls(pa.schema(fields))

    @classmethod
    def from_dataframe(cls, df: Any) -> ArrowSchema:
        """Infer schema from a pandas or Dask DataFrame.

        Note: For Dask DataFrames, this computes the schema from metadata
        without triggering execution.
        """
        import dask.dataframe as dd

        if isinstance(df, dd.DataFrame):
            # Use Dask metadata only — no compute
            arrow_schema = _pandas_meta_to_arrow_schema(df._meta)
        else:
            arrow_schema = _pandas_meta_to_arrow_schema(df)
        return cls(arrow_schema)

    @classmethod
    def empty(cls) -> ArrowSchema:
        """Build an empty schema (zero columns)."""
        return cls(pa.schema([]))

    # ------------------------------------------------------------------
    # Schema operations
    # ------------------------------------------------------------------

    def equals(self, other: ArrowSchema) -> bool:
        """Check if two schemas are equal (column names and types)."""
        return self._schema.equals(other._schema)

    def contains_column(self, name: str) -> bool:
        return name in self._schema.names

    def validate_columns(self, df: Any, *, require_all: bool = True) -> list[str]:
        """Validate that a DataFrame has the expected columns.

        Returns list of missing columns. If require_all=False, only warns.
        """
        df_columns = set(df.columns)
        schema_columns = set(self._schema.names)

        if require_all:
            missing = sorted(schema_columns - df_columns)
        else:
            missing = sorted(df_columns - schema_columns)

        return missing

    # ------------------------------------------------------------------
    # Coercion
    # ------------------------------------------------------------------

    def cast_table(self, table: pa.Table) -> pa.Table:
        """Cast an Arrow Table to this schema in a single pass.

        Uses ``table.cast()`` for batch coercion instead of per-column
        pandas operations. Handles type mismatches via safe/unsafe casting.
        """
        from boti_data.db.arrow_schema_mapper import coerce_arrow_table

        return coerce_arrow_table(table, self._schema)

    def to_arrow_table(self, df: Any) -> pa.Table:
        """Convert a pandas/Dask DataFrame to an Arrow Table with this schema.

        Uses PyArrow's native conversion with type preservation.
        """
        import dask.dataframe as dd

        if isinstance(df, dd.DataFrame):
            df = df.compute()

        # Convert to Arrow first
        table = pa.Table.from_pandas(df, preserve_index=False)

        # Then cast to target schema (handles type mismatches)
        try:
            return self.cast_table(table)
        except (KeyError, TypeError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError):
            # If cast fails, return as-is — caller can handle errors
            return table

    def to_dataframe(self, table: pa.Table, *, as_pandas: bool = True) -> Any:
        """Convert an Arrow Table to pandas with proper type mapping."""
        from boti_data.db.arrow_schema_mapper import arrow_table_to_pandas

        df = arrow_table_to_pandas(table)

        # Ensure column order matches schema
        df = df[[field.name for field in self._schema if field.name in df.columns]]

        return df

    def to_pandas(self, table: pa.Table) -> pd.DataFrame:
        """Compatibility alias for callers that expect a pandas-specific helper."""
        return self.to_dataframe(table)

    def coerce_dataframe(self, df: Any) -> Any:
        """Coerce a pandas/Dask DataFrame to this schema.

        Single-pass operation that replaces the per-column ``apply_schema_map()`` loop.
        """
        import dask.dataframe as dd

        if isinstance(df, dd.DataFrame):
            # For Dask, we need to compute — Arrow coercion is not lazy
            df = df.compute()

        table = self.to_arrow_table(df)
        return self.to_dataframe(table)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_table(self, table: pa.Table, *, require_columns: bool = True) -> list[str]:
        """Validate an Arrow Table against this schema.

        Returns list of errors (empty if valid).
        """
        errors: list[str] = []

        if require_columns:
            missing = self.validate_columns(
                type("MockDF", (), {"columns": list(table.column_names)})(),
                require_all=True,
            )
            if missing:
                errors.append(f"Missing columns: {missing}")

        for field in self._schema:
            if field.name not in table.column_names:
                continue

            actual_type = table.schema.field(field.name).type
            if actual_type != field.type:
                # Check if compatible (safe cast possible)
                try:
                    table.column(field.name).cast(field.type, safe=True)
                except (pa.ArrowInvalid, pa.ArrowTypeError):
                    errors.append(
                        f"Column '{field.name}': expected {field.type}, found {actual_type} "
                        f"(incompatible types)"
                    )

        return errors

    def validate_dataframe(self, df: Any, *, require_columns: bool = True) -> list[str]:
        """Validate a pandas/Dask DataFrame against this schema.

        Returns list of errors (empty if valid).
        """
        import dask.dataframe as dd

        if isinstance(df, dd.DataFrame):
            df = df._meta  # Use metadata

        table = self.to_arrow_table(df)
        return self.validate_table(table, require_columns=require_columns)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, str]:
        """Convert to a pandas-style dtype mapping for compatibility."""
        from boti_data.db.arrow_schema_mapper import _PANDAS_DTYPE_TO_ARROW

        # Reverse mapping: Arrow → pandas dtype string
        arrow_to_pandas = {v: k for k, v in _PANDAS_DTYPE_TO_ARROW.items()}

        result = {}
        for field in self._schema:
            pandas_dtype = arrow_to_pandas.get(field.type, str(field.type))
            result[field.name] = pandas_dtype
        return result

    def __repr__(self) -> str:
        return f"ArrowSchema({len(self._schema)} columns: {self.column_names})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ArrowSchema):
            return self.equals(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple((f.name, f.type) for f in self._schema))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _pandas_dtype_to_arrow(dtype_str: str) -> pa.DataType:
    """Convert a pandas dtype string to PyArrow type."""
    from boti_data.db.arrow_schema_mapper import pandas_dtype_to_arrow

    return pandas_dtype_to_arrow(dtype_str)


def _pandas_meta_to_arrow_schema(df: Any) -> pa.Schema:
    """Extract Arrow Schema from pandas DataFrame metadata."""
    fields = []
    for col in df.columns:
        dtype = df.dtypes[col]
        arrow_type = _pandas_dtype_to_arrow(str(dtype))
        fields.append(pa.field(col, arrow_type))
    return pa.schema(fields)


# ---------------------------------------------------------------------------
# Convenience functions (drop-in replacements for schema.py functions)
# ---------------------------------------------------------------------------


def apply_arrow_schema_map(
    dataframe: Any,
    schema_map: Mapping[str, str],
    *,
    require_columns: bool = False,
) -> Any:
    """Cast a DataFrame to a schema using PyArrow single-pass coercion.

    Drop-in replacement for ``schema.apply_schema_map()`` that uses Arrow
    for coercion instead of per-column pandas operations.
    """
    arrow_schema = ArrowSchema.from_dict(schema_map)

    # Validate required columns
    if require_columns:
        missing = arrow_schema.validate_columns(
            dataframe,
            require_all=True,
        )
        if missing:
            from boti_data.schema import SchemaValidationError

            raise SchemaValidationError(f"Missing required column(s): {missing}.")

    return arrow_schema.coerce_dataframe(dataframe)


def validate_arrow_schema(
    dataframe: Any,
    expected_schema_map: Mapping[str, str],
    *,
    require_columns: bool = True,
) -> None:
    """Validate a DataFrame against an expected schema using Arrow.

    Drop-in replacement for ``schema.validate_schema()``.
    """
    arrow_schema = ArrowSchema.from_dict(expected_schema_map)
    errors = arrow_schema.validate_dataframe(
        dataframe,
        require_columns=require_columns,
    )
    if errors:
        from boti_data.schema import SchemaValidationError

        raise SchemaValidationError("Schema validation failed:\n" + "\n".join(errors))
