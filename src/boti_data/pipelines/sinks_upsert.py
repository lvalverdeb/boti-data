"""Upsert sink: write a frame into a SQL table, updating rows whose keys already exist.

Dialect-detected statement shapes:

* PostgreSQL and SQLite: ``INSERT ... ON CONFLICT (keys) DO UPDATE SET ...``
* DuckDB: ``INSERT OR REPLACE INTO ...``
* MySQL: ``INSERT ... ON DUPLICATE KEY UPDATE ...``

pandas, Arrow and polars frames are written in one transaction, in chunks;
a Dask frame is written partition by partition. The target table must exist
and have a primary key or unique constraint on ``key_columns``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pandas as pd
import sqlalchemy as sa
from boti.core.lifecycle import LifecycleCore

from boti_data._engines import is_dask_frame, is_polars_frame
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.db.sql_resource import SqlDatabaseResource
from boti_data.pipelines.sinks_common import FrameResult, SinkWriteResult, _AsyncWriteViaThreadMixin

__all__ = ["UpsertSink", "upsert_statement"]

DEFAULT_CHUNK_ROWS = 5_000


def _pandas_frames(frame: FrameResult) -> Iterator[pd.DataFrame]:
    """Yield pandas pieces of any supported frame without holding a Dask frame whole."""
    if isinstance(frame, pd.DataFrame):
        yield frame
    elif is_dask_frame(frame):
        for index in range(frame.npartitions):
            yield frame.get_partition(index).compute()
    elif is_polars_frame(frame) or hasattr(frame, "to_pandas"):
        yield frame.to_pandas()
    else:
        raise TypeError(f"UpsertSink cannot write {type(frame)!r}")


def _records(chunk: pd.DataFrame) -> list[dict[str, Any]]:
    rows = chunk.astype(object).where(chunk.notna(), None)
    return rows.to_dict(orient="records")


def _on_conflict_statement(
    table: sa.Table, dialect: str, keys: list[str], updatable: list[str]
) -> Any:
    module = __import__(f"sqlalchemy.dialects.{dialect}", fromlist=["insert"])
    stmt = module.insert(table)
    if not updatable:
        return stmt.on_conflict_do_nothing(index_elements=keys)
    return stmt.on_conflict_do_update(
        index_elements=keys, set_={c: stmt.excluded[c] for c in updatable}
    )


def _mysql_statement(table: sa.Table, dialect: str, keys: list[str], updatable: list[str]) -> Any:
    from sqlalchemy.dialects.mysql import insert as mysql_insert

    stmt = mysql_insert(table)
    return stmt.on_duplicate_key_update(**{c: stmt.inserted[c] for c in updatable or keys})


def _duckdb_statement(table: sa.Table, dialect: str, keys: list[str], updatable: list[str]) -> Any:
    columns = keys + updatable
    quoted = ", ".join(f'"{c}"' for c in columns)
    marks = ", ".join(f":{c}" for c in columns)
    return sa.text(f'INSERT OR REPLACE INTO "{table.name}" ({quoted}) VALUES ({marks})')


_STATEMENT_BUILDERS: dict[str, Callable[[sa.Table, str, list[str], list[str]], Any]] = {
    "postgresql": _on_conflict_statement,
    "sqlite": _on_conflict_statement,
    "mysql": _mysql_statement,
    "duckdb": _duckdb_statement,
}


def upsert_statement(
    table: sa.Table, dialect: str, key_columns: list[str], columns: list[str]
) -> Any:
    """The dialect's insert-or-update statement for ``columns`` keyed on ``key_columns``."""
    builder = _STATEMENT_BUILDERS.get(dialect)
    if builder is None:
        raise ValueError(f"UpsertSink does not support dialect {dialect!r}")
    updatable = [c for c in columns if c not in key_columns]
    return builder(table, dialect, list(key_columns), updatable)


class UpsertSink(_AsyncWriteViaThreadMixin, LifecycleCore):
    """Write frames into ``table`` with insert-or-update semantics on ``key_columns``."""

    def __init__(
        self,
        config: SqlDatabaseConfig | Mapping[str, Any],
        *,
        table: str,
        key_columns: list[str],
        schema: str | None = None,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
    ) -> None:
        if not key_columns:
            raise ValueError("UpsertSink needs at least one key column.")
        base = (
            config if isinstance(config, SqlDatabaseConfig) else SqlDatabaseConfig(**dict(config))
        )
        self.config = base.model_copy(update={"query_only": False})
        self.table_name = table
        self.key_columns = list(key_columns)
        self.schema = schema
        self.chunk_rows = max(1, int(chunk_rows))
        self._resource = SqlDatabaseResource(self.config)
        self._table: sa.Table | None = None
        super().__init__()

    @property
    def engine(self) -> Any:
        return self._resource.engine

    def _cleanup(self) -> None:
        self._table = None
        self._resource.close()

    def _reflected_table(self) -> sa.Table:
        if self._table is None:
            self._table = sa.Table(
                self.table_name, sa.MetaData(), schema=self.schema, autoload_with=self.engine
            )
            missing = [c for c in self.key_columns if c not in self._table.c]
            if missing:
                raise ValueError(f"key_columns {missing} not found on table {self.table_name!r}")
        return self._table

    def _write_pandas(self, conn: Any, frame: pd.DataFrame) -> int:
        table = self._reflected_table()
        columns = [c for c in frame.columns if c in table.c]
        missing_keys = [c for c in self.key_columns if c not in columns]
        if missing_keys:
            raise ValueError(f"frame lacks key columns {missing_keys}")
        statement = upsert_statement(table, self.engine.dialect.name, self.key_columns, columns)
        for start in range(0, len(frame), self.chunk_rows):
            chunk = frame.iloc[start : start + self.chunk_rows][columns]
            conn.execute(statement, _records(chunk))
        return len(frame)

    def write(self, frame: FrameResult, **options: Any) -> SinkWriteResult:
        """Upsert ``frame`` into the table.

        The ``PipelineSink`` keyword flags (``date_field``, ``write_index``,
        ``overwrite``, ``persist``) are accepted so the sink is interchangeable
        with the file sinks, and ignored: an upsert is always an in-place merge.
        """
        del options
        with self.engine.begin() as conn:
            for piece in _pandas_frames(frame):
                self._write_pandas(conn, piece)
        return SinkWriteResult(path=f"{self.engine.dialect.name}://{self.table_name}", files=())
