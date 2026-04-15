"""
Arrow adapters for DataGateway post-load transforms.

Provides Arrow-native equivalents of Dask operations like ``sort_by()``,
``drop_duplicates()``, and ``concat_tables()`` that are zero-copy or
significantly faster than their pandas/Dask counterparts.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pyarrow as pa
import pyarrow.compute as pc


# ---------------------------------------------------------------------------
# Sort
# ---------------------------------------------------------------------------

def sort_table(
    table: pa.Table,
    sort_keys: str | Sequence[tuple[str, str]],
    *,
    ascending: bool | Sequence[bool] = True,
) -> pa.Table:
    """Sort an Arrow Table by one or more columns.

    Usage::

        # Single column ascending
        sort_table(table, "created_at")

        # Multiple columns with direction
        sort_table(table, [("status", "ascending"), ("created_at", "descending")])
    """
    if isinstance(sort_keys, str):
        sort_keys = [(sort_keys, "ascending" if ascending else "descending")]

    return table.sort_by(sort_keys)


# ---------------------------------------------------------------------------
# Drop duplicates
# ---------------------------------------------------------------------------

def drop_duplicates(
    table: pa.Table,
    subset: Sequence[str] | None = None,
    *,
    keep: str = "first",
) -> pa.Table:
    """Drop duplicate rows from an Arrow Table.

    Args:
        table: Input table.
        subset: Columns to consider for deduplication. None = all columns.
        keep: Which duplicate to keep — "first", "last", or "none".
    """
    if subset is None:
        subset = table.column_names

    if not subset:
        return table

    if keep == "first":
        aggregation = "min"
    elif keep == "last":
        aggregation = "max"
    elif keep == "none":
        if len(subset) == 1:
            key_col = table.column(subset[0])
        else:
            key_col = pa.StructArray.from_arrays(
                [table.column(col) for col in subset],
                names=subset,
            )
        # Keep only rows that appear exactly once — use Arrow compute to avoid
        # materialising the column into Python objects via tolist().
        value_counts = pc.value_counts(key_col)
        singleton_values = value_counts.field("values").filter(
            pc.equal(value_counts.field("counts"), pa.scalar(1, type=pa.int64()))
        )
        if len(singleton_values) == 0:
            return table.slice(0, 0)
        mask = pc.is_in(key_col, value_set=singleton_values)
        return table.filter(mask)
    else:
        raise ValueError(f"keep must be 'first', 'last', or 'none', got {keep!r}")

    row_index_name = "__boti_row_index__"
    indexed = table.append_column(
        row_index_name,
        pa.array(range(table.num_rows), type=pa.int64()),
    )
    grouped = indexed.group_by(list(subset)).aggregate([(row_index_name, aggregation)])
    index_column_name = f"{row_index_name}_{aggregation}"
    sorted_indices = (
        pa.table({"idx": grouped[index_column_name]})
        .sort_by([("idx", "ascending")])
        .column("idx")
    )
    if len(sorted_indices) == 0:
        return table.slice(0, 0)
    return table.take(sorted_indices)


# ---------------------------------------------------------------------------
# Group by + aggregate
# ---------------------------------------------------------------------------

def group_by_aggregate(
    table: pa.Table,
    group_keys: Sequence[str],
    aggregations: dict[str, str],
) -> pa.Table:
    """Perform group-by aggregation on an Arrow Table.

    Args:
        table: Input table.
        group_keys: Columns to group by.
        aggregations: Mapping of column → aggregation function.
            Supported: "count", "sum", "mean", "min", "max", "any".

    Usage::

        group_by_aggregate(
            table,
            group_keys=["department"],
            aggregations={"salary": "sum", "age": "mean"},
        )
    """
    gb = table.group_by(group_keys)

    agg_exprs = []
    for col, fn in aggregations.items():
        if fn == "count":
            agg_exprs.append((col, "count"))
        elif fn == "sum":
            agg_exprs.append((col, "sum"))
        elif fn == "mean":
            agg_exprs.append((col, "mean"))
        elif fn == "min":
            agg_exprs.append((col, "min"))
        elif fn == "max":
            agg_exprs.append((col, "max"))
        elif fn == "any":
            agg_exprs.append((col, "any"))
        else:
            raise ValueError(f"Unsupported aggregation: {fn!r}")

    return gb.aggregate(agg_exprs)


# ---------------------------------------------------------------------------
# Concat tables (zero-copy when schemas match)
# ---------------------------------------------------------------------------

def concat_tables(
    tables: Sequence[pa.Table],
    *,
    promote_options: str = "none",
) -> pa.Table:
    """Concatenate multiple Arrow Tables.

    Uses ``pa.concat_tables()`` which is zero-copy when schemas are identical.

    Args:
        tables: Sequence of tables to concatenate.
        promote_options: Schema mismatch handling — "none", "default", "intersect".
    """
    if not tables:
        raise ValueError("Cannot concatenate empty table list")
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote_options=promote_options)


# ---------------------------------------------------------------------------
# Select columns
# ---------------------------------------------------------------------------

def select_columns(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    """Select specific columns from a table (O(1) operation)."""
    return table.select(columns)


def project_columns(
    table: pa.Table,
    expressions: dict[str, pa.ChunkedArray | pa.Scalar],
) -> pa.Table:
    """Add or replace columns via expressions.

    Usage::

        project_columns(table, {
            "full_name": pc.utf8_trim_whitespace(pc.binary_join([table["first"], table["last"]], " ")),
        })
    """
    new_names = list(table.column_names)
    for name, expr in expressions.items():
        if name in new_names:
            idx = new_names.index(name)
            table = table.set_column(idx, name, expr)
        else:
            table = table.append_column(name, expr)
    return table


# ---------------------------------------------------------------------------
# Head / limit
# ---------------------------------------------------------------------------

def head_table(table: pa.Table, n: int) -> pa.Table:
    """Return the first n rows of a table (zero-copy slice)."""
    return table.slice(0, min(n, table.num_rows))


# ---------------------------------------------------------------------------
# Apply DataFrame options (Arrow equivalents of _apply_df_options)
# ---------------------------------------------------------------------------

def apply_table_options(
    table: pa.Table,
    *,
    sort_field: str | None = None,
    sort_ascending: bool = True,
    duplicate_expr: list[str] | None = None,
    duplicate_keep: str = "first",
    group_by_expr: str | list[str] | None = None,
    group_expr: str | dict[str, str] | None = None,
    fieldnames: list[str] | None = None,
    column_names: list[str] | None = None,
) -> pa.Table:
    """Apply post-load transform options to an Arrow Table.

    Arrow-native equivalent of ``DataGateway._apply_df_options()``.
    """
    # Sort
    if sort_field:
        table = sort_table(table, [(sort_field, "ascending" if sort_ascending else "descending")])

    # Drop duplicates
    if duplicate_expr:
        table = drop_duplicates(table, subset=duplicate_expr, keep=duplicate_keep)

    if group_by_expr is not None and group_expr is not None:
        group_keys = [group_by_expr] if isinstance(group_by_expr, str) else list(group_by_expr)
        if isinstance(group_expr, str):
            aggregations = {
                name: group_expr for name in table.column_names if name not in set(group_keys)
            }
        else:
            aggregations = dict(group_expr)
        table = group_by_aggregate(table, group_keys=group_keys, aggregations=aggregations)

    # Select/rename columns
    if fieldnames:
        table = select_columns(table, fieldnames)

    if column_names:
        source_names = fieldnames if fieldnames else list(table.column_names)
        rename_map = dict(zip(source_names, column_names))
        new_names = [rename_map.get(name, name) for name in table.column_names]
        table = table.rename_columns(new_names)

    return table


# ---------------------------------------------------------------------------
# Semi-join pattern (Arrow native)
# ---------------------------------------------------------------------------

def semi_join(
    left: pa.Table,
    right: pa.Table | pa.ChunkedArray,
    on: str,
) -> pa.Table:
    """Return rows from left table where on column value appears in right.

    Arrow-native equivalent of the distributed semi-join pattern.
    """
    if isinstance(right, pa.Table):
        right_values = right.column(on)
    else:
        right_values = right

    # Filter left table
    mask = pc.is_in(left.column(on), value_set=right_values)
    return left.filter(mask)


def anti_join(
    left: pa.Table,
    right: pa.Table | pa.ChunkedArray,
    on: str,
) -> pa.Table:
    """Return rows from left table where on column value does NOT appear in right."""
    if isinstance(right, pa.Table):
        right_values = right.column(on)
    else:
        right_values = right

    mask = pc.invert(pc.is_in(left.column(on), value_set=right_values))
    return left.filter(mask)


# ---------------------------------------------------------------------------
# Date range filtering (Arrow native)
# ---------------------------------------------------------------------------

def filter_date_range(
    table: pa.Table,
    column: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pa.Table:
    """Filter table rows by date range on a timestamp column.

    Args:
        table: Input table.
        column: Timestamp column name.
        start: ISO-8601 start date (inclusive).
        end: ISO-8601 end date (inclusive).
    """
    col = table.column(column)

    if start:
        start_ts = pa.scalar(start, type=pa.timestamp("ns", tz="UTC"))
        mask = pc.greater_equal(col, start_ts)
        table = table.filter(mask)

    if end:
        end_ts = pa.scalar(end, type=pa.timestamp("ns", tz="UTC"))
        mask = pc.less_equal(col, end_ts)
        table = table.filter(mask)

    return table
