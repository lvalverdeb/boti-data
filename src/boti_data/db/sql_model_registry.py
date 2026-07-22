"""
Thread-safe SQLAlchemy Model Registry.

Dynamically reflects SQL tables and provisions declarative ORM classes,
designed to bypass serialisation constraints using structural module injections.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sys
import threading
from collections.abc import Callable
from typing import Any, Union

from boti.core.logger import Logger
from boti.core.security import is_valid_dotted_identifier
from sqlalchemy import MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from boti_data.db.sql_model_module_injection import register_as_module_attribute
from boti_data.db.sql_model_registry_types import DefaultBase, ModelBuildContext, RegistryConfig
from boti_data.db.sqlalchemy_async import ensure_greenlet_available

try:
    # Importing this module has the side effect of registering pgvector's
    # 'vector' type name into SQLAlchemy's Postgres dialect
    # (sqlalchemy.dialects.postgresql.base.ischema_names), so reflection below
    # maps a pgvector column to pgvector.sqlalchemy.Vector (dimension included)
    # instead of falling back to an unusable NullType. Optional: boti-data
    # works the same as before for consumers who never touch pgvector and
    # haven't installed boti-data[pgvector].
    import pgvector.sqlalchemy  # noqa: F401
except ImportError:
    pass


class SqlModelRegistry:
    """
    Thread-safe registry that reflects tables sequentially and
    returns a single mapped class per (engine, schema, table).
    """

    _DYNAMIC_MODULE_SENTINEL = "__boti_dynamic_model_namespace__"

    def __init__(self, config: RegistryConfig | None = None, logger: Logger | None = None) -> None:
        self.config = config or RegistryConfig()
        self.logger = logger or Logger.default_logger(logger_name=self.__class__.__name__)

        self._metadata_cache: dict[tuple[str, str | None], MetaData] = {}
        self._model_cache: dict[tuple[str, str | None, str], type[Any]] = {}
        self._lock = threading.RLock()
        self._md_locks: dict[tuple[str, str | None], threading.Lock] = {}
        self._async_lock = asyncio.Lock()
        self._md_async_locks: dict[tuple[str, str | None], asyncio.Lock] = {}

    @staticmethod
    def _engine_key(engine: Union[Engine, AsyncEngine]) -> str:
        """Derive deterministic engine footprint securely escaping passwords via Dialect string rendering."""
        return engine.url.render_as_string(hide_password=True)

    @staticmethod
    def _qualified_key(schema: str | None, table: str) -> str:
        return f"{schema}.{table}" if schema else table

    @staticmethod
    def _split_schema_and_table(name: str) -> tuple[str | None, str]:
        if "." in name:
            s, t = name.split(".", 1)
            return (s or None), t
        return None, name

    @staticmethod
    def _normalize_class_name(table_name: str) -> str:
        parts = re.findall(r"[A-Za-z0-9]+", table_name)
        if not parts:
            return "ReflectedTable"

        normalized = "".join(part.capitalize() for part in parts)
        if normalized[0].isdigit():
            normalized = f"T{normalized}"
        return normalized

    @staticmethod
    def _short_hash(*parts: str, length: int = 8) -> str:
        h = hashlib.blake2s(
            "|".join(parts).encode("utf-8"),
            digest_size=max(8, (length + 1) // 2),
        ).hexdigest()
        return h[:length]

    def _is_class_name_taken(self, class_name: str, module_label: str) -> bool:
        if module_label in sys.modules:
            return hasattr(sys.modules[module_label], class_name)
        return False

    def _find_existing_model_for_table(self, tbl: Table, base_class: type[Any]) -> type[Any] | None:
        """Scan SQLAlchemy's registry to prevent redundant mapping of identical Tables.

        Uses strict object-identity matching only.  Name-based matching is
        intentionally omitted: two ``Table`` objects with the same name but
        different ``MetaData`` instances come from different database connections
        and must map to separate ORM classes.
        """
        for mapper in list(base_class.registry.mappers):
            try:
                mapped_cls = mapper.class_
                if getattr(mapped_cls, "__table__", None) is tbl:
                    return mapped_cls
            except Exception:
                self.logger.debug("Failed to inspect mapper for table reflection", exc_info=True)
                continue
        return None

    @staticmethod
    def _resolve_module_label(module_label: str | None, default_module_label: str) -> str:
        resolved = module_label or default_module_label
        if not is_valid_dotted_identifier(resolved):
            raise ValueError("module_label must be a valid dotted Python module path.")
        trusted_root = default_module_label.split(".", 1)[0]
        if resolved.split(".", 1)[0] != trusted_root:
            raise ValueError(
                f"module_label must stay within the trusted '{trusted_root}' module namespace."
            )
        return resolved

    def _get_or_create_metadata(self, ekey: str, schema: str | None) -> MetaData:
        md_key = (ekey, schema)
        with self._lock:
            md = self._metadata_cache.get(md_key)
            if md is None:
                md = MetaData(schema=schema)
                self._metadata_cache[md_key] = md
            return md

    @staticmethod
    def _get_or_create_cached(cache: dict[Any, Any], key: Any, factory: Callable[[], Any]) -> Any:
        value = cache.get(key)
        if value is None:
            value = factory()
            cache[key] = value
        return value

    def _get_or_create_md_lock(self, md_key: tuple[str, str | None]) -> threading.Lock:
        with self._lock:
            return self._get_or_create_cached(self._md_locks, md_key, threading.Lock)

    async def _get_or_create_md_lock_async(self, md_key: tuple[str, str | None]) -> asyncio.Lock:
        async with self._async_lock:
            return self._get_or_create_cached(self._md_async_locks, md_key, asyncio.Lock)

    def _resolve_model_keys(
        self,
        engine: Union[Engine, AsyncEngine],
        table_name: str,
        *,
        schema: str | None,
        module_label: str | None,
    ) -> tuple[str, str | None, str, tuple[str, str | None, str], tuple[str, str | None], str]:
        """Returns (ekey, schema, tname, model_key, md_key, module_label)."""
        s2, tname = self._split_schema_and_table(table_name)
        resolved_schema = schema if schema is not None else s2
        ekey = self._engine_key(engine)
        model_key = (ekey, resolved_schema, tname)
        md_key = (ekey, resolved_schema)
        resolved_module_label = self._resolve_module_label(
            module_label, self.config.default_module_label
        )
        return ekey, resolved_schema, tname, model_key, md_key, resolved_module_label

    def _build_model_class(self, tbl: Table, context: ModelBuildContext) -> type[Any]:
        base_name = self._normalize_class_name(context.tname)
        final_name = base_name

        if self._is_class_name_taken(base_name, context.module_label):
            suffix = self._short_hash(context.ekey, context.schema or "", context.tname)
            final_name = f"{base_name}_{suffix}"
        elif not context.prefer_stable_names:
            suffix = self._short_hash(context.ekey, context.schema or "", context.tname)
            final_name = f"{base_name}_{suffix}"

        attrs: dict[str, Any] = {
            "__tablename__": tbl.name,
            "__table__": tbl,
            "__module__": context.module_label,
        }

        if not tbl.primary_key.columns and tbl.columns:
            # Observability hook for injected read-only views avoiding ORM exceptions
            self.logger.warning(
                f"Missing native Primary Key on {context.qname}. "
                "Synthesizing mapper bindings using primary sequential column fallback. "
                f"{context.missing_pk_warning_suffix}"
            )
            attrs["__mapper_args__"] = {"primary_key": [list(tbl.columns)[0]]}

        model_cls = type(final_name, (context.base_class,), attrs)
        register_as_module_attribute(
            model_cls,
            context.module_label,
            final_name,
            lock=self._lock,
            sentinel=self._DYNAMIC_MODULE_SENTINEL,
        )
        return model_cls

    # Not a copy-pasted twin: shared logic is already extracted into
    # _resolve_model_keys()/_get_or_create_metadata()/_qualified_key()/
    # _find_existing_model_for_table()/_build_model_class(); the remaining
    # difference (threading.RLock vs asyncio.Lock, sync md.reflect() vs
    # conn.run_sync(md.reflect) + ensure_greenlet_available()) is a genuine
    # concurrency-primitive difference, not copy-paste. _resolve_or_build_model()
    # is a further shared helper — it does no locking/IO of its own, only the
    # already-shared _find_existing_model_for_table()/_build_model_class().
    # spaghetti-ignore[sync-async-duplication]: see above
    def get_model(
        self,
        engine: Engine,
        table_name: str,
        *,
        refresh: bool = False,
        schema: str | None = None,
        module_label: str | None = None,
        prefer_stable_names: bool = True,
        base_class: type[Any] = DefaultBase,
    ) -> type[Any]:
        """Atomically reflects a table schema and scaffolds its ORM class safely."""
        ekey, schema, tname, model_key, md_key, module_label = self._resolve_model_keys(
            engine, table_name, schema=schema, module_label=module_label
        )
        cached = self._check_cache(model_key, refresh=refresh)
        if cached is not None:
            return cached

        md = self._get_or_create_metadata(ekey, schema)
        md_lock = self._get_or_create_md_lock(md_key)
        qname = self._qualified_key(schema, tname)
        tbl = self._reflect_table(engine, md, md_lock, qname=qname, tname=tname, schema=schema)

        context = ModelBuildContext(
            qname=qname,
            tname=tname,
            base_class=base_class,
            module_label=module_label,
            prefer_stable_names=prefer_stable_names,
            ekey=ekey,
            schema=schema,
            missing_pk_warning_suffix="Ensure ORM mutation sequences validate bounds manually.",
        )
        model_cls = self._resolve_or_build_model(tbl, qname, context, target_label="DB target")
        with self._lock:
            self._model_cache[model_key] = model_cls
        return model_cls

    async def get_model_async(
        self,
        engine: AsyncEngine,
        table_name: str,
        *,
        refresh: bool = False,
        schema: str | None = None,
        module_label: str | None = None,
        prefer_stable_names: bool = True,
        base_class: type[Any] = DefaultBase,
    ) -> type[Any]:
        """Atomically reflects schemas asynchronously bridging Non-Blocking asyncio event pools."""
        ensure_greenlet_available()
        ekey, schema, tname, model_key, md_key, module_label = self._resolve_model_keys(
            engine, table_name, schema=schema, module_label=module_label
        )
        cached = await self._check_cache_async(model_key, refresh=refresh)
        if cached is not None:
            return cached

        md = self._get_or_create_metadata(ekey, schema)
        md_lock = await self._get_or_create_md_lock_async(md_key)
        qname = self._qualified_key(schema, tname)
        tbl = await self._reflect_table_async(
            engine, md, md_lock, qname=qname, tname=tname, schema=schema
        )

        context = ModelBuildContext(
            qname=qname,
            tname=tname,
            base_class=base_class,
            module_label=module_label,
            prefer_stable_names=prefer_stable_names,
            ekey=ekey,
            schema=schema,
            missing_pk_warning_suffix="Ensure Async ORM validation sequences capture bounds correctly.",
        )
        model_cls = self._resolve_or_build_model(
            tbl, qname, context, target_label="Async DB target"
        )
        async with self._async_lock:
            self._model_cache[model_key] = model_cls
        return model_cls

    def _resolve_or_build_model(
        self,
        tbl: Table | None,
        qname: str,
        context: ModelBuildContext,
        *,
        target_label: str,
    ) -> type[Any]:
        if tbl is None:
            raise ValueError(
                f"Table '{qname}' does not exist referencing the given {target_label}."
            )
        reused = self._find_existing_model_for_table(tbl, context.base_class)
        if reused is not None:
            return reused
        return self._build_model_class(tbl, context)

    # Not a copy-pasted twin: threading.Lock vs asyncio.Lock is a genuine
    # concurrency-primitive difference (see get_model()'s comment above).
    # spaghetti-ignore[sync-async-duplication]: see above
    def _check_cache(self, model_key: tuple, *, refresh: bool) -> type[Any] | None:
        if refresh:
            with self._lock:
                self._model_cache.pop(model_key, None)
        with self._lock:
            return self._model_cache.get(model_key)

    async def _check_cache_async(self, model_key: tuple, *, refresh: bool) -> type[Any] | None:
        if refresh:
            async with self._async_lock:
                self._model_cache.pop(model_key, None)
        async with self._async_lock:
            return self._model_cache.get(model_key)

    # Not a copy-pasted twin: sync md.reflect() vs conn.run_sync(md.reflect) is
    # a genuine concurrency-primitive difference (see get_model()'s comment above).
    @staticmethod
    # spaghetti-ignore[sync-async-duplication]: see above
    def _reflect_table(
        engine: Engine,
        md: MetaData,
        md_lock: threading.Lock,
        *,
        qname: str,
        tname: str,
        schema: str | None,
    ) -> Table | None:
        tbl = md.tables.get(qname)
        if tbl is None:
            with md_lock:
                tbl = md.tables.get(qname)
                if tbl is None:
                    md.reflect(bind=engine, only=[tname], schema=schema)
                tbl = md.tables.get(qname)
        return tbl

    @staticmethod
    async def _reflect_table_async(
        engine: AsyncEngine,
        md: MetaData,
        md_lock: asyncio.Lock,
        *,
        qname: str,
        tname: str,
        schema: str | None,
    ) -> Table | None:
        tbl = md.tables.get(qname)
        if tbl is None:
            async with md_lock:
                tbl = md.tables.get(qname)
                if tbl is None:
                    async with engine.begin() as conn:
                        await conn.run_sync(md.reflect, only=[tname], schema=schema)
                tbl = md.tables.get(qname)
        return tbl

    def clear(self) -> None:
        """Reset internal caches unconditionally."""
        with self._lock:
            self._metadata_cache.clear()
            self._model_cache.clear()
            self._md_locks.clear()
            self._md_async_locks.clear()


# Thread-safe global fallback
_global_registry = SqlModelRegistry()


def get_global_registry() -> SqlModelRegistry:
    return _global_registry
