from __future__ import annotations

import os
from typing import Any, Dict, Optional, Type

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.engine import url as sqlalchemy_url
from sqlalchemy.exc import NoSuchModuleError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool, Pool, QueuePool, StaticPool

from boti_data.db.sql_config import SqlDatabaseConfig, WorkerSqlConfig

_CONNECTION_URL_DRIVER_ALIASES = {
    "mysql+pymsql": "mysql+pymysql",
}


def _normalize_connection_url(connection_url: str) -> str:
    parsed = sqlalchemy_url.make_url(connection_url)
    normalized_driver = _CONNECTION_URL_DRIVER_ALIASES.get(parsed.drivername, parsed.drivername)
    if normalized_driver == parsed.drivername:
        return parsed.render_as_string(hide_password=False)
    return parsed.set(drivername=normalized_driver).render_as_string(hide_password=False)


def _parse_connection_url(
    connection_url: str,
    *,
    consumer_name: str,
) -> sqlalchemy_url.URL:
    normalized_url = _normalize_connection_url(connection_url)
    parsed = sqlalchemy_url.make_url(normalized_url)
    try:
        parsed.get_dialect()
    except NoSuchModuleError as exc:
        raise SQLAlchemyError(
            f"{consumer_name} received unsupported SQLAlchemy dialect '{parsed.drivername}'. "
            "Check the driver name in the connection URL."
        ) from exc
    return parsed


def _build_driver_example(parsed_url: sqlalchemy_url.URL, *, async_mode: bool) -> str:
    """Return a driver example that matches the requested execution mode."""
    backend = parsed_url.get_backend_name()
    if async_mode:
        async_examples = {
            "mysql": "mysql+asyncmy://",
            "postgresql": "postgresql+asyncpg://",
            "sqlite": "sqlite+aiosqlite://",
        }
        return async_examples.get(backend, f"{backend}+<async-driver>://")

    sync_examples = {
        "mysql": "mysql+pymysql://",
        "postgresql": "postgresql+psycopg://",
        "sqlite": "sqlite://",
    }
    return sync_examples.get(backend, f"{backend}://")


def _render_normalized_url(connection_url: str, *, hide_password: bool) -> str:
    parsed = sqlalchemy_url.make_url(_normalize_connection_url(connection_url))
    return parsed.set(query={k: v for k, v in parsed.query.items()}).render_as_string(
        hide_password=hide_password
    )


def _validate_sync_driver_url(
    connection_url: str,
    *,
    consumer_name: str = "SqlDatabaseResource",
) -> sqlalchemy_url.URL:
    parsed = _parse_connection_url(connection_url, consumer_name=consumer_name)
    if parsed.get_dialect().is_async:
        suggested_driver = _build_driver_example(parsed, async_mode=False)
        raise SQLAlchemyError(
            f"{consumer_name} only supports synchronous SQLAlchemy drivers. "
            f"Received async dialect '{parsed.drivername}'. "
            "Use AsyncSqlDatabaseResource instead, or switch to a synchronous DSN such as "
            f"'{suggested_driver}...'."
        )
    return parsed


def _validate_async_driver_url(
    connection_url: str,
    *,
    consumer_name: str = "AsyncSqlDatabaseResource",
) -> sqlalchemy_url.URL:
    parsed = _parse_connection_url(connection_url, consumer_name=consumer_name)
    if not parsed.get_dialect().is_async:
        suggested_driver = _build_driver_example(parsed, async_mode=True)
        raise SQLAlchemyError(
            f"{consumer_name} requires an asynchronous SQLAlchemy driver. "
            f"Received synchronous dialect '{parsed.drivername}'. "
            f"Use SqlDatabaseResource instead, or switch to an async DSN such as '{suggested_driver}...'."
        )
    return parsed


def _build_sync_engine_kwargs(
    config: SqlDatabaseConfig,
    *,
    poolclass_override: Optional[Type[Pool]] = None,
) -> Dict[str, Any]:
    poolclass = poolclass_override or config.poolclass
    kwargs: Dict[str, Any] = {
        "pool_recycle": config.pool_recycle,
        "pool_pre_ping": config.pool_pre_ping,
        "poolclass": poolclass,
        "connect_args": config.connect_args,
        "execution_options": config.execution_options,
    }
    if poolclass not in (NullPool, StaticPool):
        kwargs.update(
            {
                "pool_size": config.pool_size,
                "max_overflow": config.max_overflow,
                "pool_timeout": config.pool_timeout,
            }
        )
    return kwargs


def _build_async_engine_kwargs(config: SqlDatabaseConfig) -> Dict[str, Any]:
    actual_poolclass = config.poolclass
    kwargs: Dict[str, Any] = {
        "pool_recycle": config.pool_recycle,
        "pool_pre_ping": config.pool_pre_ping,
        "connect_args": config.connect_args,
        "execution_options": config.execution_options,
    }
    if actual_poolclass is not QueuePool:
        kwargs["poolclass"] = actual_poolclass
    if actual_poolclass not in (NullPool, StaticPool):
        kwargs.update(
            {
                "pool_size": config.pool_size,
                "max_overflow": config.max_overflow,
                "pool_timeout": config.pool_timeout,
            }
        )
    return kwargs


def _build_sync_engine_key(config: SqlDatabaseConfig) -> tuple:
    normalized_url = _render_normalized_url(
        config.connection_url.get_secret_value(),
        hide_password=True,
    )
    key_parts = ["sync", str(normalized_url), config.poolclass, config.query_only]
    if config.poolclass not in (NullPool, StaticPool):
        key_parts.extend([config.pool_size, config.max_overflow, config.pool_timeout])
    key_parts.extend([config.pool_recycle, config.pool_pre_ping])
    return tuple(key_parts)


def _build_async_engine_key(config: SqlDatabaseConfig) -> tuple:
    normalized_url = _render_normalized_url(
        config.connection_url.get_secret_value(),
        hide_password=True,
    )
    actual_poolclass = config.poolclass
    key_poolclass = "async_default_queue" if actual_poolclass is QueuePool else actual_poolclass

    key_parts = ["async", str(normalized_url), key_poolclass, config.query_only]
    if actual_poolclass not in (NullPool, StaticPool):
        key_parts.extend([config.pool_size, config.max_overflow, config.pool_timeout])
    key_parts.extend([config.pool_recycle, config.pool_pre_ping])
    return tuple(key_parts)


def _coerce_worker_sql_config(
    config: SqlDatabaseConfig | WorkerSqlConfig,
) -> WorkerSqlConfig:
    if isinstance(config, WorkerSqlConfig):
        return config
    return WorkerSqlConfig.from_database_config(config)


def _get_worker_engine_identity(config: SqlDatabaseConfig | WorkerSqlConfig) -> str:
    worker_config = _coerce_worker_sql_config(config)
    if worker_config.connection_env_var is not None:
        connection_identity = f"env:{worker_config.connection_env_var}"
    else:
        assert worker_config.connection_url is not None
        connection_identity = _render_normalized_url(
            worker_config.connection_url.get_secret_value(),
            hide_password=True,
        )

    return "|".join(
        [
            "worker-sync",
            connection_identity,
            f"query_only={int(worker_config.query_only)}",
        ]
    )


def _resolve_worker_connection_url(config: WorkerSqlConfig) -> str:
    if config.connection_env_var is not None:
        value = os.environ.get(config.connection_env_var)
        if not value:
            raise SQLAlchemyError(
                f"Worker DB connection environment variable '{config.connection_env_var}' is not set."
            )
        return value

    if config.connection_url is None:
        raise SQLAlchemyError("Worker SQL configuration is missing a connection URL.")
    return config.connection_url.get_secret_value()


def _build_worker_sync_engine_kwargs(config: WorkerSqlConfig) -> Dict[str, Any]:
    return {
        "pool_recycle": config.pool_recycle,
        "pool_pre_ping": config.pool_pre_ping,
        "poolclass": NullPool,
        "connect_args": config.connect_args,
        "execution_options": config.execution_options,
    }


def _build_worker_async_engine_kwargs(config: WorkerSqlConfig) -> Dict[str, Any]:
    return {
        "pool_recycle": config.pool_recycle,
        "pool_pre_ping": config.pool_pre_ping,
        "poolclass": NullPool,
        "connect_args": config.connect_args,
        "execution_options": config.execution_options,
    }


def _create_worker_async_engine(config: SqlDatabaseConfig | WorkerSqlConfig) -> AsyncEngine:
    worker_config = _coerce_worker_sql_config(config)
    connection_url = _normalize_connection_url(_resolve_worker_connection_url(worker_config))
    parsed = _validate_async_driver_url(
        connection_url,
        consumer_name="AsyncSqlPartitionedLoader",
    )
    if worker_config.query_only:
        _validate_query_only_support(parsed)

    return create_async_engine(
        connection_url,
        **_build_worker_async_engine_kwargs(worker_config),
    )


def _create_worker_sync_engine(config: SqlDatabaseConfig | WorkerSqlConfig) -> Engine:
    worker_config = _coerce_worker_sql_config(config)
    connection_url = _normalize_connection_url(_resolve_worker_connection_url(worker_config))
    parsed = _validate_sync_driver_url(
        connection_url,
        consumer_name="SqlPartitionedLoader",
    )
    if worker_config.query_only:
        _validate_query_only_support(parsed)

    engine = create_engine(
        connection_url,
        **_build_worker_sync_engine_kwargs(worker_config),
    )
    if worker_config.query_only:
        _configure_query_only_engine(engine, parsed)
    return engine


def _validate_query_only_support(parsed_url: sqlalchemy_url.URL) -> None:
    backend = parsed_url.get_backend_name()

    if backend in {"postgresql", "mysql"}:
        return

    if backend == "sqlite":
        database = parsed_url.database or ""
        mode = str(parsed_url.query.get("mode", "")).lower()
        uri = str(parsed_url.query.get("uri", "")).lower()
        if database == ":memory:":
            raise SQLAlchemyError(
                "query_only=True requires native database-enforced read-only access. "
                "SQLite in-memory databases cannot enforce read-only mode. "
                "Use query_only=False for tests, or a file DSN with 'mode=ro&uri=true'."
            )
        if mode == "ro" and uri in {"1", "true", "yes"}:
            return
        raise SQLAlchemyError(
            "SQLite query_only mode requires a file DSN with native read-only flags, "
            "for example 'sqlite:///file:/absolute/path/to/db?mode=ro&uri=true'."
        )

    raise SQLAlchemyError(
        f"query_only=True requires native read-only enforcement, but backend '{backend}' "
        "is not currently configured for that in SqlDatabaseResource."
    )


def _configure_query_only_engine(engine: Engine, parsed_url: sqlalchemy_url.URL) -> None:
    backend = parsed_url.get_backend_name()

    if backend == "postgresql":

        @event.listens_for(engine, "connect")
        def _set_postgresql_read_only(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("SET default_transaction_read_only = on")
            finally:
                cursor.close()

        return

    if backend == "mysql":

        @event.listens_for(engine, "connect")
        def _set_mysql_read_only(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("SET SESSION TRANSACTION READ ONLY")
            finally:
                cursor.close()

        return

    if backend == "sqlite":
        return
