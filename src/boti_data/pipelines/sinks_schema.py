"""Schema evolution for Parquet sink appends.

A dataset's schema may only widen over time: columns can be added and types
promoted (``int32`` to ``int64``, ``float32`` to ``float64``, integer to
float, ``null`` to anything, timestamp unit widening). An append whose type
cannot be merged with the existing column's type (``string`` into ``int64``,
say) is refused with :class:`SchemaNarrowingError` before anything is written.
Nothing is rewritten in place: new files carry the widened schema and readers
unify the files' schemas on read (see ``ParquetDataResource.load_files``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["SchemaNarrowingError", "evolved_write_schema", "frame_schema", "unify_dataset_schemas"]


class SchemaNarrowingError(ValueError):
    """An append would narrow or change a column's type incompatibly."""

    def __init__(self, column: str, existing: pa.DataType, incoming: pa.DataType) -> None:
        self.column = column
        self.existing = existing
        self.incoming = incoming
        super().__init__(
            f"Parquet sink append would change column {column!r} from {existing} to "
            f"{incoming}, which cannot be merged. Datasets may only widen: add columns "
            "or promote types; write the frame with the existing type, or overwrite."
        )


def _first_parquet_file(fs: Any, target_path: str) -> str | None:
    files = sorted(fs.glob(f"{target_path.rstrip('/')}/**/*.parquet"))
    return files[0] if files else None


def existing_dataset_schema(fs: Any, target_path: str) -> pa.Schema | None:
    """Schema of the dataset already at *target_path*, or ``None`` if there is none."""
    file = _first_parquet_file(fs, target_path)
    if file is None:
        return None
    with fs.open(file, "rb") as handle:
        return pq.read_schema(handle).remove_metadata()


def unify_dataset_schemas(schemas: Sequence[pa.Schema]) -> pa.Schema:
    """Unified (widened) schema of *schemas*; :class:`SchemaNarrowingError` if impossible."""
    bare = [schema.remove_metadata() for schema in schemas]
    try:
        return pa.unify_schemas(bare, promote_options="permissive")
    except (pa.ArrowTypeError, pa.ArrowInvalid) as exc:
        column, existing, incoming = _first_conflict(bare)
        raise SchemaNarrowingError(column, existing, incoming) from exc


def _first_conflict(schemas: Sequence[pa.Schema]) -> tuple[str, pa.DataType, pa.DataType]:
    first = schemas[0]
    for later in schemas[1:]:
        for field in later:
            index = first.get_field_index(field.name)
            if index < 0:
                continue
            try:
                pa.unify_schemas(
                    [pa.schema([first[index]]), pa.schema([field])], promote_options="permissive"
                )
            except (pa.ArrowTypeError, pa.ArrowInvalid):
                return field.name, first[index].type, field.type
    raise AssertionError("unify_schemas failed but no single column conflicts")


def frame_schema(ddf: Any, *, write_index: bool) -> pa.Schema:
    """Arrow schema a Dask frame will be written with."""
    # spaghetti-ignore[encapsulation-violation]: _meta is dask's documented frame metadata
    return pa.Schema.from_pandas(ddf._meta, preserve_index=write_index)


def evolved_write_schema(
    fs: Any,
    target_path: str,
    incoming: pa.Schema,
    *,
    partition_on: Sequence[str] | None,
) -> pa.Schema | None:
    """Schema to write an append with, or ``None`` when no dataset exists yet.

    Partition columns live in directory names, not in the files, so they are
    left out of the comparison and restored on the result for the writer. A
    column the existing dataset has but the frame lacks is simply absent from
    the new files; readers unify the files' schemas and fill it with nulls.
    """
    existing = existing_dataset_schema(fs, target_path)
    if existing is None:
        return None
    partition_fields = [
        incoming.field(name) for name in (partition_on or []) if name in incoming.names
    ]
    data_incoming = incoming.remove_metadata()
    for field in partition_fields:
        data_incoming = data_incoming.remove(data_incoming.get_field_index(field.name))
    unified = unify_dataset_schemas([existing, data_incoming])
    # A column the frame lacks stays out of its files (readers fill nulls);
    # a column it has is written at the unified, possibly promoted, type.
    projected = pa.schema([unified.field(name) for name in data_incoming.names])
    for field in partition_fields:
        projected = projected.append(field)
    return projected
