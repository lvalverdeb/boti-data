from __future__ import annotations

import logging
import threading
from typing import Any

from boti.core.logger import Logger
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine

_log = logging.getLogger(__name__)


class EngineRegistry:
    """Thread-safe cache for SQLAlchemy engines constructed with identical footprints."""

    _lock = threading.RLock()
    _registry: dict[tuple, dict[str, Any]] = {}

    @classmethod
    def _check_registered(cls, key: tuple) -> Any | None:
        """Returns the cached engine (incrementing ref_count) if already registered."""
        with cls._lock:
            wrapper = cls._registry.get(key)
            if wrapper is not None:
                wrapper["ref_count"] += 1
                return wrapper["engine"]
        return None

    @classmethod
    def _register_or_flag_duplicate(
        cls, key: tuple, engine: Any, *, is_async: bool = False
    ) -> tuple[Any, bool, Any | None]:
        """Registers ``engine`` under ``key`` unless another thread beat us to it.

        Returns ``(engine_to_use, was_cached, duplicate_engine_to_dispose)``.
        """
        with cls._lock:
            wrapper = cls._registry.get(key)
            if wrapper is not None:
                wrapper["ref_count"] += 1
                return wrapper["engine"], True, engine
            entry: dict[str, Any] = {"engine": engine, "ref_count": 1}
            if is_async:
                entry["is_async"] = True
            cls._registry[key] = entry
            return engine, False, None

    @classmethod
    def _log_duplicate_dispose_error(cls, key: tuple) -> None:
        _log.debug("Error disposing duplicate engine for key %s", key, exc_info=True)

    # Not a copy-pasted twin: shared logic already lives in _check_registered()/
    # _register_or_flag_duplicate()/_log_duplicate_dispose_error(); the remaining
    # difference is the irreducible create_engine() vs create_async_engine() and
    # dispose() vs await dispose() calls.
    @classmethod
    # spaghetti-ignore[sync-async-duplication]
    def get_or_create(cls, key: tuple, url: str, **kwargs: Any) -> tuple[Engine, bool]:
        cached = cls._check_registered(key)
        if cached is not None:
            return cached, True

        engine = create_engine(url, **kwargs)
        result_engine, was_cached, duplicate = cls._register_or_flag_duplicate(key, engine)
        if duplicate is not None:
            try:
                duplicate.dispose()
            except Exception:
                cls._log_duplicate_dispose_error(key)
        return result_engine, was_cached

    @classmethod
    async def get_or_create_async(cls, key: tuple, url: str, **kwargs: Any) -> tuple[Any, bool]:
        cached = cls._check_registered(key)
        if cached is not None:
            return cached, True

        engine = create_async_engine(url, **kwargs)
        result_engine, was_cached, duplicate = cls._register_or_flag_duplicate(
            key, engine, is_async=True
        )
        if duplicate is not None:
            # Guarded the same way as the sync path's duplicate.dispose() a
            # few lines up: a disposal failure here must not propagate and
            # fail an otherwise-successful get_or_create_async() call. This
            # was previously unguarded — the sync/async twins had drifted.
            try:
                await duplicate.dispose()
            except Exception:
                cls._log_duplicate_dispose_error(key)
        return result_engine, was_cached

    @classmethod
    def _decrement_and_maybe_pop(cls, key: tuple, logger: Logger | None = None) -> Any | None:
        """Decrements ref_count; pops and returns the engine to dispose if it hit zero."""
        with cls._lock:
            if key not in cls._registry:
                return None
            wrapper = cls._registry[key]
            wrapper["ref_count"] -= 1

            if logger:
                try:
                    logger.debug("DB Engine ref count: %s for %s", wrapper["ref_count"], key)
                except Exception:
                    _log.debug("Failed to log engine ref count for key %s", key, exc_info=True)

            if wrapper["ref_count"] <= 0:
                cls._registry.pop(key, None)
                return wrapper["engine"]
        return None

    @classmethod
    def _log_disposed(cls, key: tuple, logger: Logger | None) -> None:
        if logger:
            try:
                logger.debug("Disposed DB Engine for key %s", key)
            except Exception:
                _log.debug("Failed to log engine disposal for key %s", key, exc_info=True)

    @classmethod
    def _log_dispose_error(cls, key: tuple) -> None:
        _log.debug("Error disposing engine for key %s", key, exc_info=True)

    @classmethod
    def _log_release_error(cls, key: tuple, *, is_async: bool) -> None:
        kind = "async " if is_async else ""
        _log.debug("Error during %sengine registry release for key %s", kind, key, exc_info=True)

    # Not a copy-pasted twin: shared logic already lives in
    # _decrement_and_maybe_pop()/_log_disposed()/_log_dispose_error()/
    # _log_release_error(); the remaining difference is the irreducible
    # engine.dispose() vs await engine.dispose() call.
    @classmethod
    # spaghetti-ignore[sync-async-duplication]
    def release(cls, key: tuple, logger: Logger | None = None) -> None:
        if cls._lock is None:
            return

        try:
            engine = cls._decrement_and_maybe_pop(key, logger)
            if engine is not None:
                try:
                    engine.dispose()
                    cls._log_disposed(key, logger)
                except Exception:
                    cls._log_dispose_error(key)
        except Exception:
            cls._log_release_error(key, is_async=False)

    @classmethod
    async def release_async(cls, key: tuple, logger: Logger | None = None) -> None:
        if cls._lock is None:
            return

        try:
            engine = cls._decrement_and_maybe_pop(key, logger)
            if engine is not None:
                try:
                    await engine.dispose()
                    cls._log_disposed(key, logger)
                except Exception:
                    cls._log_dispose_error(key)
        except Exception:
            cls._log_release_error(key, is_async=True)
