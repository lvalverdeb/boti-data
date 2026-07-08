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
import types
from typing import Any, Union

from boti.core.logger import Logger
from boti.core.security import is_valid_dotted_identifier
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import DeclarativeBase

from boti_data.db.sqlalchemy_async import ensure_greenlet_available


class RegistryConfig(BaseModel):
    """Configuration mapping establishing deterministic registry footprints."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    default_module_label: str = Field(default="boti_data.dynamic_models")

    @field_validator("default_module_label")
    @classmethod
    def validate_default_module_label(cls, value: str) -> str:
        """Restrict registry module targets to valid module paths."""
        if not is_valid_dotted_identifier(value):
            raise ValueError("default_module_label must be a valid dotted Python module path.")
        return value


class DefaultBase(DeclarativeBase):
    """Shared declarative base for prototype ORM models."""
    pass


class SqlModelRegistry:
    """
    Thread-safe registry that reflects tables sequentially and
    returns a single mapped class per (engine, schema, table).
    """

    _DYNAMIC_MODULE_SENTINEL = "__boti_dynamic_model_namespace__"

    def __init__(
        self,
        config: RegistryConfig | None = None,
        logger: Logger | None = None
    ) -> None:
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

    def _get_or_create_md_lock(self, md_key: tuple[str, str | None]) -> threading.Lock:
        with self._lock:
            lock = self._md_locks.get(md_key)
            if lock is None:
                lock = threading.Lock()
                self._md_locks[md_key] = lock
            return lock

    async def _get_or_create_md_lock_async(self, md_key: tuple[str, str | None]) -> asyncio.Lock:
        async with self._async_lock:
            lock = self._md_async_locks.get(md_key)
            if lock is None:
                lock = asyncio.Lock()
                self._md_async_locks[md_key] = lock
            return lock

    def _register_as_module_attribute(self, cls: type, module_name: str, class_name: str) -> None:
        """
        Injects the class into sys.modules maintaining Pickling safety across processes.
        """
        parts = module_name.split(".")
        with self._lock:
            existing = sys.modules.get(module_name)
            if existing is not None and not getattr(
                existing,
                self._DYNAMIC_MODULE_SENTINEL,
                False,
            ):
                raise ValueError(
                    "module_label must target an unused or registry-managed dynamic module namespace."
                )

            for index in range(1, len(parts) + 1):
                current_name = ".".join(parts[:index])
                current_module = sys.modules.get(current_name)
                if current_module is None:
                    current_module = types.ModuleType(current_name)
                    current_module.__package__ = current_name.rpartition(".")[0]
                    current_module.__path__ = []  # type: ignore[attr-defined]
                    setattr(current_module, self._DYNAMIC_MODULE_SENTINEL, True)
                    sys.modules[current_name] = current_module
                elif index == len(parts) and not getattr(
                    current_module,
                    self._DYNAMIC_MODULE_SENTINEL,
                    False,
                ):
                    raise ValueError(
                        "module_label must target an unused or registry-managed dynamic module namespace."
                    )

                if index > 1:
                    parent_name = ".".join(parts[: index - 1])
                    child_name = parts[index - 1]
                    setattr(sys.modules[parent_name], child_name, current_module)

            setattr(sys.modules[module_name], class_name, cls)

    def get_model(
        self,
        engine: Engine,
        table_name: str,
        *,
        refresh: bool = False,
        schema: str | None = None,
        module_label: str | None = None,
        prefer_stable_names: bool = True,
        base_class: type[Any] = DefaultBase
    ) -> type[Any]:
        """
        Atomically reflects a table schema and scaffolds its ORM class safely.
        """
        s2, tname = self._split_schema_and_table(table_name)
        schema = schema if schema is not None else s2
        ekey = self._engine_key(engine)
        model_key = (ekey, schema, tname)
        md_key = (ekey, schema)
        module_label = self._resolve_module_label(module_label, self.config.default_module_label)

        if refresh:
            with self._lock:
                self._model_cache.pop(model_key, None)

        with self._lock:
            m = self._model_cache.get(model_key)
            if m is not None:
                return m

        md = self._get_or_create_metadata(ekey, schema)
        md_lock = self._get_or_create_md_lock(md_key)
        qname = self._qualified_key(schema, tname)

        tbl = md.tables.get(qname)
        if tbl is None:
            with md_lock:
                tbl = md.tables.get(qname)
                if tbl is None:
                    md.reflect(bind=engine, only=[tname], schema=schema)
                tbl = md.tables.get(qname)

        if tbl is None:
            raise ValueError(f"Table '{qname}' does not exist referencing the given DB target.")

        reused = self._find_existing_model_for_table(tbl, base_class)
        if reused is not None:
            with self._lock:
                self._model_cache[model_key] = reused
            return reused

        base_name = self._normalize_class_name(tname)
        final_name = base_name

        if self._is_class_name_taken(base_name, module_label):
            suffix = self._short_hash(ekey, schema or "", tname)
            final_name = f"{base_name}_{suffix}"
        elif not prefer_stable_names:
            suffix = self._short_hash(ekey, schema or "", tname)
            final_name = f"{base_name}_{suffix}"

        attrs: dict[str, Any] = {
            "__tablename__": tbl.name,
            "__table__": tbl,
            "__module__": module_label,
        }

        if not tbl.primary_key.columns and tbl.columns:
            # Observability hook for injected read-only views avoiding ORM exceptions
            self.logger.warning(
                f"Missing native Primary Key on {qname}. "
                "Synthesizing mapper bindings using primary sequential column fallback. "
                "Ensure ORM mutation sequences validate bounds manually."
            )
            attrs["__mapper_args__"] = {"primary_key": [list(tbl.columns)[0]]}

        model_cls = type(final_name, (base_class,), attrs)

        self._register_as_module_attribute(model_cls, module_label, final_name)

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
        base_class: type[Any] = DefaultBase
    ) -> type[Any]:
        """Atomically reflects schemas asynchronously bridging Non-Blocking asyncio event pools."""
        ensure_greenlet_available()
        s2, tname = self._split_schema_and_table(table_name)
        schema = schema if schema is not None else s2
        ekey = self._engine_key(engine)
        model_key = (ekey, schema, tname)
        md_key = (ekey, schema)
        module_label = self._resolve_module_label(module_label, self.config.default_module_label)

        if refresh:
            async with self._async_lock:
                self._model_cache.pop(model_key, None)

        async with self._async_lock:
            m = self._model_cache.get(model_key)
            if m is not None: return m

        md = self._get_or_create_metadata(ekey, schema)
        md_lock = await self._get_or_create_md_lock_async(md_key)
        qname = self._qualified_key(schema, tname)

        tbl = md.tables.get(qname)
        if tbl is None:
            async with md_lock:
                tbl = md.tables.get(qname)
                if tbl is None:
                    async with engine.begin() as conn:
                        await conn.run_sync(md.reflect, only=[tname], schema=schema)
                tbl = md.tables.get(qname)

        if tbl is None:
            raise ValueError(f"Table '{qname}' does not exist referencing the given Async DB target.")

        reused = self._find_existing_model_for_table(tbl, base_class)
        if reused is not None:
            async with self._async_lock:
                self._model_cache[model_key] = reused
            return reused

        base_name = self._normalize_class_name(tname)
        final_name = base_name

        if self._is_class_name_taken(base_name, module_label):
            suffix = self._short_hash(ekey, schema or "", tname)
            final_name = f"{base_name}_{suffix}"
        elif not prefer_stable_names:
            suffix = self._short_hash(ekey, schema or "", tname)
            final_name = f"{base_name}_{suffix}"

        attrs: dict[str, Any] = {
            "__tablename__": tbl.name,
            "__table__": tbl,
            "__module__": module_label,
        }

        if not tbl.primary_key.columns and tbl.columns:
            self.logger.warning(
                f"Missing native Primary Key on {qname}. "
                "Synthesizing mapper bindings using primary sequential column fallback. "
                "Ensure Async ORM validation sequences capture bounds correctly."
            )
            attrs["__mapper_args__"] = {"primary_key": [list(tbl.columns)[0]]}

        model_cls = type(final_name, (base_class,), attrs)
        self._register_as_module_attribute(model_cls, module_label, final_name)

        async with self._async_lock:
            self._model_cache[model_key] = model_cls
        return model_cls

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
