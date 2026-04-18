from __future__ import annotations

import asyncio
import threading
from typing import Any

from boti.core.logger import Logger
from boti.core.managed_resource import ManagedResource
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker

from boti_data.db.engine_registry import EngineRegistry
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.db.sql_engine import (
    _build_async_engine_key,
    _build_async_engine_kwargs,
    _build_sync_engine_key,
    _build_sync_engine_kwargs,
    _configure_query_only_engine,
    _normalize_connection_url,
    _render_normalized_url,
    _validate_async_driver_url,
    _validate_query_only_support,
    _validate_sync_driver_url,
)
from boti_data.db.sql_readonly import ReadOnlyAsyncSession, ReadOnlySession
from boti_data.db.sqlalchemy_async import ensure_greenlet_available


class SqlDatabaseResource(ManagedResource):
    """Managed context object bridging SqlDatabaseConfig and the EngineRegistry."""

    def __init__(self, config: SqlDatabaseConfig, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        self.config: SqlDatabaseConfig = config

        self._engine: Engine | None = None
        self._engine_view: Any | None = None
        self._session_factory: sessionmaker | None = None
        self._engine_key: tuple | None = None

        self._initialize_sql_environment()

    def _get_engine_key(self) -> tuple:
        return _build_sync_engine_key(self.config)

    def _initialize_sql_environment(self) -> None:
        try:
            connection_url = _normalize_connection_url(self.config.connection_url.get_secret_value())
            parsed = _validate_sync_driver_url(connection_url)
            if self.config.query_only:
                _validate_query_only_support(parsed)
            self._engine_key = self._get_engine_key()

            engine, reused = EngineRegistry.get_or_create(
                self._engine_key,
                connection_url,
                **_build_sync_engine_kwargs(self.config),
            )
            if self.config.query_only and not reused:
                _configure_query_only_engine(engine, parsed)

            self._engine = engine
            self._engine_view = engine
            self._session_factory = sessionmaker(
                bind=self._engine,
                expire_on_commit=False,
                class_=ReadOnlySession if self.config.query_only else Session,  # type: ignore[name-defined]
            )

            if self.debug:
                safe_url = _render_normalized_url(
                    self.config.connection_url.get_secret_value(),
                    hide_password=True,
                )
                action = "Reused cached" if reused else "Spawned new"
                self.logger.debug(f"{action} SQLAlchemy instance connected against {safe_url}")

        except Exception as exc:
            self.logger.error(f"SQL Backend initialization failed: {exc}")
            raise SQLAlchemyError(f"DB Engine creation failed: {exc}") from exc

    def _cleanup(self) -> None:
        super()._cleanup()
        if self._engine_key and self._engine:
            EngineRegistry.release(self._engine_key, logger=self.logger)
            self._engine = None
            self._engine_view = None

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state.pop("_engine", None)
        state.pop("_engine_view", None)
        state.pop("_session_factory", None)
        state.pop("_engine_key", None)
        return state

    def _restore_runtime_state(self) -> None:
        self._engine = None
        self._engine_view = None
        self._session_factory = None
        self._engine_key = None
        if not self._is_closed:
            self._initialize_sql_environment()

    @property
    def engine(self) -> Any:
        self._assert_open()
        if self._engine_view is None:
            raise RuntimeError("Engine not ready")
        return self._engine_view

    @property
    def session(self) -> sessionmaker:
        self._assert_open()
        if self._session_factory is None:
            raise RuntimeError("Factory not initialized")
        return self._session_factory


class AsyncSqlDatabaseResource:
    """Asynchronous context manager providing reflection/query-only SQLAlchemy access."""

    def __init__(self, config: SqlDatabaseConfig) -> None:
        self.config = config
        self.logger = Logger.default_logger(logger_name=self.__class__.__name__)
        self._engine: AsyncEngine | None = None
        self._engine_view: Any | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._engine_key: tuple | None = None
        self._is_open = False
        self._is_closed = False
        self._closing = False
        self._state_lock = threading.RLock()
        self._aclose_lock = asyncio.Lock()

    def _get_engine_key(self) -> tuple:
        return _build_async_engine_key(self.config)

    def _assert_not_closed(self) -> None:
        with self._state_lock:
            if self._is_closed or self._closing:
                raise RuntimeError(f"{self.__class__.__name__} is closed")

    @property
    def closed(self) -> bool:
        with self._state_lock:
            return self._is_closed

    async def __aenter__(self) -> AsyncSqlDatabaseResource:
        self._assert_not_closed()
        with self._state_lock:
            if self._is_open:
                return self

        connection_url = _normalize_connection_url(self.config.connection_url.get_secret_value())
        parsed = _validate_async_driver_url(connection_url)
        ensure_greenlet_available()
        if self.config.query_only:
            _validate_query_only_support(parsed)
        self._engine_key = self._get_engine_key()

        engine, reused = await EngineRegistry.get_or_create_async(
            self._engine_key,
            connection_url,
            **_build_async_engine_kwargs(self.config),
        )
        if self.config.query_only and not reused:
            _configure_query_only_engine(engine.sync_engine, parsed)

        with self._state_lock:
            if self._is_closed or self._closing:
                # Resource was closed while connecting; release the engine immediately.
                key = self._engine_key
            else:
                key = None
                self._engine = engine
                self._engine_view = engine
                self._session_factory = async_sessionmaker(
                    bind=self._engine,
                    expire_on_commit=False,
                    class_=ReadOnlyAsyncSession if self.config.query_only else AsyncSession,
                )
                self._is_open = True
                return self

        if key is not None:
            await EngineRegistry.release_async(key, logger=self.logger)
        raise RuntimeError(f"{self.__class__.__name__} is closed")

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose(suppress_errors=exc_type is not None)

    async def aclose(self, *, suppress_errors: bool = False) -> None:
        async with self._aclose_lock:
            with self._state_lock:
                if self._is_closed or self._closing:
                    return
                self._closing = True
                key = self._engine_key
                has_engine = self._engine is not None

            try:
                if key is not None and has_engine:
                    await EngineRegistry.release_async(key, logger=self.logger)
            except Exception:
                self.logger.error(
                    f"Error during {self.__class__.__name__}.aclose()",
                    exc_info=self.config.debug,
                )
                if not suppress_errors:
                    raise
            finally:
                with self._state_lock:
                    self._engine = None
                    self._engine_view = None
                    self._session_factory = None
                    self._engine_key = None
                    self._is_open = False
                    self._is_closed = True
                    self._closing = False

    @property
    def engine(self) -> Any:
        if not self._is_open:
            raise RuntimeError("Async DB is not bound.")
        return self._engine_view

    @property
    def session(self) -> async_sessionmaker[AsyncSession]:
        if not self._is_open:
            raise RuntimeError("Async DB is not bound.")
        return self._session_factory  # type: ignore[return-value]
