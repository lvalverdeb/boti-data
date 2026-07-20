"""LRU cache of resolved configured-mode select statements for DataGateway."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from boti_data.db import SqlDatabaseResource

from .loaders import reflect_and_select, reflect_and_select_async


class ConfiguredSelectCache:
    """LRU cache of resolved (model, statement) pairs for configured-mode loads.

    Sync and async lookups use separate underlying caches (their
    ``reflect_and_select``/``reflect_and_select_async`` calls are not
    interchangeable). Touch-on-read LRU ordering; evicts the oldest entry
    once past *maxsize*.
    """

    def __init__(self, *, maxsize: int = 128) -> None:
        self._maxsize = maxsize
        self._sync_cache: OrderedDict[tuple[str, ...], tuple[Any, Any]] = OrderedDict()
        self._async_cache: OrderedDict[tuple[str, ...], tuple[Any, Any]] = OrderedDict()

    @staticmethod
    def _cache_key(db_columns: list[str] | None) -> tuple[str, ...]:
        return tuple(db_columns or ())

    def _lru_get(
        self, cache: OrderedDict[tuple[str, ...], tuple[Any, Any]], cache_key: tuple[str, ...]
    ) -> tuple[Any, Any] | None:
        """LRU touch-on-read: bumps *cache_key* to most-recently-used if present."""
        cached = cache.get(cache_key)
        if cached is not None:
            cache.move_to_end(cache_key)
        return cached

    def _lru_put(
        self,
        cache: OrderedDict[tuple[str, ...], tuple[Any, Any]],
        cache_key: tuple[str, ...],
        value: tuple[Any, Any],
    ) -> None:
        """Inserts *value*, evicting the least-recently-used entry over *_maxsize*."""
        cache[cache_key] = value
        if len(cache) > self._maxsize:
            cache.popitem(last=False)

    # Not a copy-pasted twin: shared _cache_key()/_lru_get()/_lru_put() are
    # already extracted; the remaining difference is reflect_and_select(...)
    # vs await reflect_and_select_async(...) plus a sync-only defensive assert.
    # spaghetti-ignore[sync-async-duplication]
    def get(
        self, resource: Any, table: str | None, db_columns: list[str] | None
    ) -> tuple[Any, Any]:
        cache_key = self._cache_key(db_columns)
        cached = self._lru_get(self._sync_cache, cache_key)
        if cached is not None:
            return cached
        assert isinstance(resource, SqlDatabaseResource)
        result = reflect_and_select(resource, table, db_columns)  # type: ignore[arg-type]
        self._lru_put(self._sync_cache, cache_key, result)
        return result

    async def get_async(
        self, table: str | None, resource: Any, db_columns: list[str] | None
    ) -> tuple[Any, Any]:
        cache_key = self._cache_key(db_columns)
        cached = self._lru_get(self._async_cache, cache_key)
        if cached is not None:
            return cached
        result = await reflect_and_select_async(resource, table, db_columns)
        self._lru_put(self._async_cache, cache_key, result)
        return result
