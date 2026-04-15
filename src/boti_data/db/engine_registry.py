from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine

from boti.core.logger import Logger

_log = logging.getLogger(__name__)


class EngineRegistry:
    """Thread-safe cache for SQLAlchemy engines constructed with identical footprints."""

    _lock = threading.RLock()
    _registry: Dict[tuple, Dict[str, Any]] = {}

    @classmethod
    def get_or_create(cls, key: tuple, url: str, **kwargs: Any) -> Tuple[Engine, bool]:
        with cls._lock:
            if key in cls._registry:
                wrapper = cls._registry[key]
                wrapper["ref_count"] += 1
                return wrapper["engine"], True

        engine = create_engine(url, **kwargs)

        with cls._lock:
            wrapper = cls._registry.get(key)
            if wrapper is not None:
                wrapper["ref_count"] += 1
                try:
                    engine.dispose()
                except Exception:
                    _log.debug("Error disposing duplicate engine for key %s", key, exc_info=True)
                return wrapper["engine"], True

            cls._registry[key] = {"engine": engine, "ref_count": 1}
            return engine, False

    @classmethod
    async def get_or_create_async(cls, key: tuple, url: str, **kwargs: Any) -> Tuple[Any, bool]:
        with cls._lock:
            if key in cls._registry:
                wrapper = cls._registry[key]
                wrapper["ref_count"] += 1
                return wrapper["engine"], True

        engine = create_async_engine(url, **kwargs)

        with cls._lock:
            wrapper = cls._registry.get(key)
            if wrapper is not None:
                wrapper["ref_count"] += 1
                await engine.dispose()
                return wrapper["engine"], True

            cls._registry[key] = {"engine": engine, "ref_count": 1, "is_async": True}
            return engine, False

    @classmethod
    def release(cls, key: tuple, logger: Optional[Logger] = None) -> None:
        if cls._lock is None:
            return

        try:
            with cls._lock:
                if key in cls._registry:
                    wrapper = cls._registry[key]
                    wrapper["ref_count"] -= 1

                    if logger:
                        try:
                            logger.debug(f"DB Engine ref count: {wrapper['ref_count']} for {key}")
                        except Exception:
                            pass

                    if wrapper["ref_count"] <= 0:
                        try:
                            wrapper["engine"].dispose()
                            if logger:
                                try:
                                    logger.debug(f"Disposed DB Engine for key {key}")
                                except Exception:
                                    pass
                        finally:
                            cls._registry.pop(key, None)
        except Exception:
            _log.debug("Error during engine registry release for key %s", key, exc_info=True)

    @classmethod
    async def release_async(cls, key: tuple, logger: Optional[Logger] = None) -> None:
        if cls._lock is None:
            return

        engine_to_dispose = None
        try:
            with cls._lock:
                if key in cls._registry:
                    wrapper = cls._registry[key]
                    wrapper["ref_count"] -= 1
                    if wrapper["ref_count"] <= 0:
                        engine_to_dispose = wrapper["engine"]
                        cls._registry.pop(key, None)

            if engine_to_dispose:
                await engine_to_dispose.dispose()
        except Exception:
            _log.debug("Error during async engine registry release for key %s", key, exc_info=True)
