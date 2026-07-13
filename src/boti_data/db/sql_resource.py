from __future__ import annotations

from typing import Any

from boti.core.lifecycle import LifecycleCore
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
                self.logger.debug("%s SQLAlchemy instance connected against %s", action, safe_url)

        except Exception as exc:
            self.logger.error("SQL Backend initialization failed: %s", exc)
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


class AsyncSqlDatabaseResource(LifecycleCore):
    """Asynchronous context manager providing reflection/query-only SQLAlchemy access.

    Async-only by design: real resource acquisition (the engine, session
    factory) happens in ``__aenter__``, not ``__init__``, since it requires
    awaiting ``EngineRegistry.get_or_create_async()``. There is deliberately
    no sync ``close()`` support — see ``_cleanup()`` — because releasing a
    bound engine requires ``await EngineRegistry.release_async()``; no sync
    path can do that safely. No pickle support either: even with
    LifecycleCore's own locks stripped, the SQLAlchemy ``AsyncEngine``/
    ``async_sessionmaker`` attributes below aren't picklable, and unlike the
    sync resources in this module there's no "reopen by DSN" pattern that
    would make reconstructing one on the other end meaningful.
    """

    def __init__(self, config: SqlDatabaseConfig) -> None:
        self.config = config
        self.logger = Logger.default_logger(logger_name=self.__class__.__name__)
        self._engine: AsyncEngine | None = None
        self._engine_view: Any | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._engine_key: tuple | None = None
        super().__init__()

    def _get_engine_key(self) -> tuple:
        return _build_async_engine_key(self.config)

    async def __aenter__(self) -> AsyncSqlDatabaseResource:
        await super().__aenter__()
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

        self._engine = engine
        self._engine_view = engine
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            class_=ReadOnlyAsyncSession if self.config.query_only else AsyncSession,
        )
        return self

    def _cleanup(self) -> None:
        if self._engine is not None:
            raise RuntimeError(
                f"{self.__class__.__name__} only supports async cleanup once bound — "
                "the underlying engine is refcounted and can only be released via an "
                "awaited coroutine. Use 'await resource.aclose()' or 'async with', not close()."
            )
        # Never entered __aenter__ (or already closed): nothing to release.

    async def _acleanup(self) -> None:
        # LifecycleCore's aclose() barrier makes this run at most once per
        # instance, even under concurrent callers — previously, aclose()
        # hand-rolled its own idempotency check ("if self._engine_key and
        # self._engine") with an await in between the check and the reset,
        # a real TOCTOU race: two concurrent aclose() calls on the same
        # instance could both pass the check before either reset state,
        # each issuing their own EngineRegistry.release_async() for what
        # should be one logical release. Confirmed harmful empirically when
        # the engine is shared (ref_count > 1, i.e. a sibling resource also
        # holds a reference): the double-release disposes the engine while
        # the sibling still believes it holds a valid, live reference.
        if self._engine_key and self._engine:
            await EngineRegistry.release_async(self._engine_key, logger=self.logger)
        self._engine = None
        self._engine_view = None
        self._session_factory = None
        self._engine_key = None

    @property
    def engine(self) -> Any:
        self._assert_open()
        if self._engine_view is None:
            raise RuntimeError("Async DB is not bound.")
        return self._engine_view

    @property
    def session(self) -> async_sessionmaker[AsyncSession]:
        self._assert_open()
        if self._session_factory is None:
            raise RuntimeError("Async DB is not bound.")
        return self._session_factory  # type: ignore[return-value]
