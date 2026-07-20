from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = [
    "DataFrameLike",
    "SchemaValidationError",
    "normalize_dtype_alias",
    "normalize_schema_map",
    "infer_schema_map",
    "apply_schema_map",
    "validate_schema",
    "align_frames_for_join",
]

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa

_logger = logging.getLogger(__name__)

DataFrameLike = pd.DataFrame | dd.DataFrame

_CANONICAL_DTYPE_ALIASES = {
    "int64": "Int64",
    "int64[pyarrow]": "Int64",
    "int32": "Int32",
    "int32[pyarrow]": "Int32",
    "float64": "Float64",
    "float64[pyarrow]": "Float64",
    "double[pyarrow]": "Float64",
    "bool": "boolean",
    "boolean": "boolean",
    "boolean[pyarrow]": "boolean",
    "string": "string",
    "string[python]": "string",
    "string[pyarrow]": "string",
    "large_string[pyarrow]": "string",
    "datetime64[ns, utc]": "datetime64[ns, UTC]",
    "datetime64[us, utc]": "datetime64[ns, UTC]",
    "datetime64[ms, utc]": "datetime64[ns, UTC]",
    "timestamp[ns, tz=utc][pyarrow]": "datetime64[ns, UTC]",
    "timestamp[us, tz=utc][pyarrow]": "datetime64[ns, UTC]",
    "timestamp[ms, tz=utc][pyarrow]": "datetime64[ns, UTC]",
    "datetime64[ns]": "datetime64[ns]",
    "datetime64[us]": "datetime64[ns]",
    "datetime64[ms]": "datetime64[ns]",
    "timestamp[ns][pyarrow]": "datetime64[ns]",
    "timestamp[us][pyarrow]": "datetime64[ns]",
    "timestamp[ms][pyarrow]": "datetime64[ns]",
    "category": "category",
    "object": "object",
}

_BOOLEAN_LITERAL_MAP = {
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "yes": True,
    "no": False,
    "y": True,
    "n": False,
    "t": True,
    "f": False,
}

_PANDAS_DATETIME_DTYPE_RE = re.compile(r"^datetime64\[(?P<unit>[^,\]]+)(?:,\s*(?P<tz>[^\]]+))?\]$")
_PYARROW_TIMESTAMP_DTYPE_RE = re.compile(
    r"^timestamp\[(?P<unit>[^,\]]+)(?:,\s*tz=(?P<tz>[^\]]+))?\]\[pyarrow\]$"
)


class SchemaValidationError(TypeError):
    """Raised when a DataFrame does not satisfy an expected schema contract."""


def _normalize_datetime_match(match: re.Match[str] | None, dtype_str: str) -> str | None:
    """Resolve a datetime/timestamp regex match to a canonical dtype, or None if no match."""
    if not match:
        return None
    timezone = match.group("tz")
    if timezone is not None:
        return "datetime64[ns, UTC]" if timezone == "utc" else dtype_str
    return "datetime64[ns]"


_DATETIME_DTYPE_PATTERNS = (_PANDAS_DATETIME_DTYPE_RE, _PYARROW_TIMESTAMP_DTYPE_RE)


def normalize_dtype_alias(dtype: Any) -> str:
    """Normalize pandas/Dask/PyArrow dtype strings to a canonical comparison form."""
    dtype_str = str(dtype).strip()
    lowered = dtype_str.lower()
    if lowered in _CANONICAL_DTYPE_ALIASES:
        return _CANONICAL_DTYPE_ALIASES[lowered]
    for pattern in _DATETIME_DTYPE_PATTERNS:
        result = _normalize_datetime_match(pattern.match(lowered), dtype_str)
        if result is not None:
            return result
    return dtype_str


def normalize_schema_map(schema_map: Mapping[str, str]) -> dict[str, str]:
    """Normalize an expected schema map to the canonical dtype alias form."""
    return {column: normalize_dtype_alias(dtype) for column, dtype in schema_map.items()}


def infer_schema_map(
    dataframe: DataFrameLike,
    *,
    columns: Sequence[str] | None = None,
) -> dict[str, str]:
    """Infer the normalized dtype map for the given DataFrame."""
    selected_columns = columns or list(dataframe.columns)
    missing = [column for column in selected_columns if column not in dataframe.columns]
    if missing:
        raise SchemaValidationError(f"Missing column(s) while inferring schema: {missing}.")

    return {column: normalize_dtype_alias(dataframe.dtypes[column]) for column in selected_columns}


def _drop_timezone_partition(series: pd.Series) -> pd.Series:
    try:
        return series.dt.tz_localize(None)
    except (AttributeError, TypeError):
        return series


def _coerce_boolean_partition(series: pd.Series) -> pd.Series:
    as_string = series.astype("string")
    stripped = as_string.str.strip()
    normalized = stripped.str.lower()
    coerced = normalized.map(_BOOLEAN_LITERAL_MAP)
    unresolved = coerced.isna() & normalized.notna()
    if unresolved.any():
        numeric = pd.to_numeric(normalized[unresolved], errors="coerce")
        coerced.loc[unresolved] = numeric.map(
            lambda value: pd.NA if pd.isna(value) else bool(int(value))
        )
    return pd.Series(coerced, index=series.index, dtype="boolean")


def _coerce_dt_utc(series: Any, is_dask: bool) -> Any:
    fn = dd.to_datetime if is_dask else pd.to_datetime
    return fn(series, errors="coerce", utc=True)


def _coerce_dt(series: Any, is_dask: bool) -> Any:
    if is_dask:
        converted = dd.to_datetime(series, errors="coerce")
        return converted.map_partitions(
            _drop_timezone_partition,
            meta=(series.name, "datetime64[ns]"),
        )
    return _drop_timezone_partition(pd.to_datetime(series, errors="coerce"))


def _coerce_numeric(series: Any, target_dtype: str, is_dask: bool) -> Any:
    fn = dd.to_numeric if is_dask else pd.to_numeric
    return fn(series, errors="coerce").astype(target_dtype)


_EXACT_DTYPE_COERCERS: dict[str, Callable[[Any, bool], Any]] = {
    "datetime64[ns, UTC]": _coerce_dt_utc,
    "datetime64[ns]": _coerce_dt,
    "boolean": lambda series, is_dask: (
        series.map_partitions(_coerce_boolean_partition, meta=(series.name, "boolean"))
        if is_dask
        else _coerce_boolean_partition(series)
    ),
    "string": lambda series, _is_dask: series.astype("string"),
    "category": lambda series, _is_dask: series.astype("category"),
}


def _coerce_series(series: Any, target_dtype: str, *, is_dask: bool) -> Any:
    if target_dtype in {"Int64", "Int32", "Float64"}:
        return _coerce_numeric(series, target_dtype, is_dask)
    coercer = _EXACT_DTYPE_COERCERS.get(target_dtype)
    if coercer is not None:
        return coercer(series, is_dask)
    return series.astype(target_dtype)


def apply_schema_map(
    dataframe: DataFrameLike,
    schema_map: Mapping[str, str],
    *,
    require_columns: bool = False,
    use_arrow: bool = True,
) -> DataFrameLike:
    """Cast the provided DataFrame columns to a canonical schema."""
    normalized_schema = normalize_schema_map(schema_map)
    missing = [column for column in normalized_schema if column not in dataframe.columns]
    if missing and require_columns:
        raise SchemaValidationError(f"Missing required column(s): {missing}.")

    if use_arrow and not isinstance(dataframe, dd.DataFrame):
        from boti_data.arrow_schema import ArrowSchema

        try:
            arrow_schema = ArrowSchema.from_dict(normalized_schema)
            table = pa.Table.from_pandas(dataframe, preserve_index=False)
            table = arrow_schema.cast_table(table)
            return arrow_schema.to_pandas(table)
        except (KeyError, TypeError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError):
            _logger.debug("ArrowSchema cast failed, falling back to per-column coercion")

    is_dask = isinstance(dataframe, dd.DataFrame)
    transforms = {}
    for column, target_dtype in normalized_schema.items():
        if column not in dataframe.columns:
            continue
        transforms[column] = _coerce_series(
            dataframe[column],
            target_dtype,
            is_dask=is_dask,
        )

    if not transforms:
        return dataframe
    return dataframe.assign(**transforms)


def validate_schema(
    dataframe: DataFrameLike,
    expected_schema_map: Mapping[str, str],
    *,
    require_columns: bool = True,
) -> None:
    """Validate that the current DataFrame dtypes match the expected schema."""
    normalized_expected = normalize_schema_map(expected_schema_map)
    errors: list[str] = []

    for column, expected_dtype in normalized_expected.items():
        if column not in dataframe.columns:
            if require_columns:
                errors.append(f"Missing column '{column}'.")
            continue

        current_dtype = normalize_dtype_alias(dataframe.dtypes[column])
        if current_dtype != expected_dtype:
            errors.append(
                f"Column '{column}': expected '{expected_dtype}', found '{dataframe.dtypes[column]}'."
            )

    if errors:
        raise SchemaValidationError("Schema validation failed:\n" + "\n".join(errors))


def align_frames_for_join(
    left: DataFrameLike,
    right: DataFrameLike,
    join_schema_map: Mapping[str, str],
) -> tuple[DataFrameLike, DataFrameLike]:
    """Normalize and validate both DataFrames against a shared join-key schema."""
    normalized_schema = normalize_schema_map(join_schema_map)
    left_aligned = apply_schema_map(left, normalized_schema, require_columns=True)
    right_aligned = apply_schema_map(right, normalized_schema, require_columns=True)
    validate_schema(left_aligned, normalized_schema, require_columns=True)
    validate_schema(right_aligned, normalized_schema, require_columns=True)
    return left_aligned, right_aligned
