"""
Typed catalog for named SQL and filesystem connection profiles.
"""

from __future__ import annotations

import functools
import re
import threading
from pathlib import Path
from typing import Optional

import fsspec
import pyarrow.fs as pafs

from boti.core.filesystem import FilesystemAdapter, FilesystemConfig, create_filesystem
from boti_data.db.sql_config import SqlDatabaseConfig
from boti_data.db.sql_resource import AsyncSqlDatabaseResource, SqlDatabaseResource

_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class ConnectionCatalog:
    """Named registry for typed connection profiles and runtime adapters."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sql_configs: dict[str, SqlDatabaseConfig] = {}
        self._filesystem_configs: dict[str, FilesystemConfig] = {}
        self._filesystem_adapters: dict[str, FilesystemAdapter] = {}

    @staticmethod
    def _validate_name(name: str) -> str:
        normalized = name.strip()
        if not normalized or not _PROFILE_NAME_PATTERN.fullmatch(normalized):
            raise ValueError(
                "Connection profile names must contain only letters, digits, dots, underscores, or hyphens."
            )
        return normalized

    def register_sql(self, name: str, config: SqlDatabaseConfig) -> SqlDatabaseConfig:
        profile_name = self._validate_name(name)
        with self._lock:
            self._sql_configs[profile_name] = config
        return config

    def load_sql(
        self,
        name: str,
        prefix: str,
        *,
        env_file: Optional[str | Path] = None,
        **overrides: object,
    ) -> SqlDatabaseConfig:
        config = SqlDatabaseConfig.from_env_prefix(prefix, env_file=env_file, **overrides)
        return self.register_sql(name, config)

    def sql_config(self, name: str) -> SqlDatabaseConfig:
        profile_name = self._validate_name(name)
        with self._lock:
            try:
                return self._sql_configs[profile_name]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown SQL profile '{profile_name}'. Available: {sorted(self._sql_configs)}"
                ) from exc

    def create_sql_resource(self, name: str) -> SqlDatabaseResource:
        return SqlDatabaseResource(self.sql_config(name))

    def create_async_sql_resource(self, name: str) -> AsyncSqlDatabaseResource:
        return AsyncSqlDatabaseResource(self.sql_config(name))

    def register_filesystem(self, name: str, config: FilesystemConfig) -> FilesystemConfig:
        profile_name = self._validate_name(name)
        with self._lock:
            self._filesystem_configs[profile_name] = config
            self._filesystem_adapters.pop(profile_name, None)
        return config

    def load_filesystem(
        self,
        name: str,
        prefix: str,
        *,
        env_file: Optional[str | Path] = None,
        **overrides: object,
    ) -> FilesystemConfig:
        config = FilesystemConfig.from_env_prefix(prefix, env_file=env_file, **overrides)
        return self.register_filesystem(name, config)

    def filesystem_config(self, name: str) -> FilesystemConfig:
        profile_name = self._validate_name(name)
        with self._lock:
            try:
                return self._filesystem_configs[profile_name]
            except KeyError as exc:
                raise KeyError(
                    f"Unknown filesystem profile '{profile_name}'. Available: {sorted(self._filesystem_configs)}"
                ) from exc

    def filesystem_adapter(self, name: str) -> FilesystemAdapter:
        profile_name = self._validate_name(name)
        with self._lock:
            adapter = self._filesystem_adapters.get(profile_name)
            if adapter is None:
                adapter = FilesystemAdapter(self.filesystem_config(profile_name))
                self._filesystem_adapters[profile_name] = adapter
            return adapter

    def filesystem(self, name: str) -> fsspec.AbstractFileSystem:
        return self.filesystem_adapter(name).get_filesystem()

    def pyarrow_filesystem(self, name: str) -> tuple[pafs.FileSystem, str]:
        return self.filesystem_adapter(name).get_pyarrow_filesystem()

    def invalidate_filesystem(self, name: str) -> None:
        self.filesystem_adapter(name).invalidate()

    def make_filesystem_factory(self, name: str):
        return functools.partial(create_filesystem, self.filesystem_config(name))
