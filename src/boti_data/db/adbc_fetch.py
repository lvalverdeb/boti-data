"""ADBC partition fetches, reachable only behind the read-only guards.

ADBC drivers hand back Arrow directly from the database protocol, skipping
the DBAPI row objects that :mod:`boti_data.db.arrow_fetch` has to transpose.
They sit outside SQLAlchemy, so this module re-applies what the SQLAlchemy
path guarantees before a partition is fetched through them:

* only with ``query_only=True``: the worker config is validated by
  :func:`boti_data.db.sql_engine._validate_query_only_support` first, so a
  SQLite DSN carries ``mode=ro`` and the backend enforces read-only natively;
* the same session statement the SQLAlchemy engine issues on connect
  (``SET default_transaction_read_only = on``, ``PRAGMA query_only``);
* only for the partition statements the planner compiled from ``Select``
  constructs, never for raw SQL text.

Backends without an ADBC driver (MySQL, ClickHouse) and writable
connections keep the SQLAlchemy path untouched. A driver error inside ADBC
falls back to the SQLAlchemy path for that partition, with a warning.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from sqlalchemy.engine import make_url

from boti_data.db.arrow_fetch import _cast_table, _check_columns
from boti_data.db.arrow_schema_mapper import build_arrow_schema_from_meta_dtypes
from boti_data.db.sql_engine import _resolve_worker_connection_url, _validate_query_only_support

if TYPE_CHECKING:
    from sqlalchemy.engine.url import URL

    from boti_data.db.partitioned_types import SqlPartitionSpec
    from boti_data.db.sql_config import WorkerSqlConfig

__all__ = ["ADBC_BACKENDS", "adbc_eligible", "fetch_partition_table", "open_read_only"]

_logger = logging.getLogger(__name__)

_POSTGRESQL = "postgresql"
_SQLITE = "sqlite"

# backend -> (driver module, session read-only statement, connect keyword arguments)
ADBC_BACKENDS: dict[str, tuple[str, str, dict[str, Any]]] = {
    _POSTGRESQL: (
        "adbc_driver_postgresql",
        "SET default_transaction_read_only = on",
        {"autocommit": True},
    ),
    _SQLITE: ("adbc_driver_sqlite", "PRAGMA query_only = 1", {}),
}

# One pass: an escaped percent is consumed before a placeholder can start inside it.
_PYFORMAT = re.compile(r"%%|%\((\w+)\)s")


def _backend_of(config: WorkerSqlConfig) -> URL:
    return make_url(_resolve_worker_connection_url(config))


def driver_installed(backend: str) -> bool:
    """Whether the ADBC driver package for *backend* is importable."""
    entry = ADBC_BACKENDS.get(backend)
    return entry is not None and importlib.util.find_spec(entry[0]) is not None


def adbc_eligible(config: WorkerSqlConfig | None) -> bool:
    """ADBC applies only to read-only workers whose backend has an installed driver."""
    if config is None or not (config.adbc and config.query_only):
        return False
    return driver_installed(_backend_of(config).get_backend_name())


def adbc_uri(url: URL) -> str:
    """The driver's own connection URI for a SQLAlchemy URL already validated read-only.

    PostgreSQL's ADBC driver speaks libpq URIs, so the SQLAlchemy driver suffix
    goes. SQLite's takes the same ``file:`` URI, keeping ``mode=ro``.
    """
    backend = url.get_backend_name()
    if backend == _SQLITE:
        return f"{url.database}?mode=ro"
    return url.set(drivername=backend).render_as_string(hide_password=False)


def _dollar_numbered(sql: str, params: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
    order: list[str] = []

    def _number(match: re.Match[str]) -> str:
        if match.group(1) is None:
            return "%"
        order.append(match.group(1))
        return f"${len(order)}"

    rewritten = _PYFORMAT.sub(_number, sql)
    return rewritten, tuple(params[name] for name in order)


def positional(sql: str, params: Any, backend: str) -> tuple[str, tuple[Any, ...] | None]:
    """Rewrite the SQLAlchemy driver's paramstyle into the ADBC driver's.

    The planner compiles partitions for the SQLAlchemy driver: psycopg's
    ``%(name)s`` (with ``%%`` for a literal percent), pysqlite's ``?``. ADBC
    PostgreSQL takes ``$1..$n``; ADBC SQLite takes ``?`` as it is.
    """
    if backend == _POSTGRESQL and isinstance(params, dict):
        return _dollar_numbered(sql, params)
    return sql, tuple(params) if isinstance(params, (list, tuple)) else None


def _connect(config: WorkerSqlConfig) -> Any:
    url = _backend_of(config)
    _validate_query_only_support(url)
    module_name, read_only_statement, connect_kwargs = ADBC_BACKENDS[url.get_backend_name()]
    dbapi = importlib.import_module(f"{module_name}.dbapi")
    connection = dbapi.connect(adbc_uri(url), **connect_kwargs)
    with connection.cursor() as cursor:
        cursor.execute(read_only_statement)
    return connection


@contextmanager
def open_read_only(config: WorkerSqlConfig) -> Iterator[Any]:
    """One ADBC connection for the duration of the block, then closed.

    Worker engines run on ``NullPool``: a connection per fetch, released as
    soon as the fetch ends, so open connections never exceed the fetch gate's
    limit. ADBC connections follow the same rule rather than lingering per
    thread.
    """
    connection = _connect(config)
    try:
        yield connection
    finally:
        connection.close()


def fetch_partition_table(
    config: WorkerSqlConfig, partition: SqlPartitionSpec, meta_dtypes: dict[str, str]
) -> pa.Table:
    """Run one compiled partition through ADBC and cast it to the plan's schema."""
    sql, params = positional(
        partition.sql, partition.params, _backend_of(config).get_backend_name()
    )
    with open_read_only(config) as connection, connection.cursor() as cursor:
        cursor.execute(sql, params) if params else cursor.execute(sql)
        table = cursor.fetch_arrow_table()
    schema = build_arrow_schema_from_meta_dtypes(meta_dtypes)
    _check_columns(table.column_names, schema)
    return _cast_table(table.select(schema.names), schema)


def driver_error_type() -> type[BaseException]:
    """The ADBC driver manager's base error, for the fallback in the executor."""
    return importlib.import_module("adbc_driver_manager").Error
