"""
Typed catalog for named SQL and filesystem connection profiles.
"""

from __future__ import annotations

import functools
import re
import threading
from pathlib import Path

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
        env_file: str | Path | None = None,
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
        env_file: str | Path | None = None,
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


class S3Catalog:
    """Local helper for S3-compatible servers configured from a prefixed .env block.

    Expected keys are PREFIX + field_name.upper(), for example:

        MINIO_FS_TYPE=s3
        MINIO_FS_PATH=my-bucket/raw/events
        MINIO_FS_KEY=...
        MINIO_FS_SECRET=...
        MINIO_FS_ENDPOINT=https://minio.local:9000
        MINIO_FS_REGION=us-east-1
        MINIO_FS_VERIFY_SSL=false
    """

    def __init__(
        self,
        prefix: str,
        *,
        env_file: str | Path = ".env",
        max_attempts: int = 3,
        retry_base_delay: float = 0.5,
        **overrides,
    ) -> None:
        self.prefix = prefix
        self.env_file = Path(env_file)
        self.max_attempts = max_attempts
        self.retry_base_delay = retry_base_delay

        self.config = FilesystemConfig.from_env_prefix(
            prefix,
            env_file=self.env_file,
            **overrides,
        )
        if self.config.fs_type not in {"s3", "s3a"}:
            raise ValueError(
                f"{prefix!r} resolved to fs_type={self.config.fs_type!r}; expected 's3' or 's3a'."
            )

        self.adapter = FilesystemAdapter(self.config)

    @property
    def storage_path(self) -> str:
        return self.adapter.storage_path

    def fs(self):
        return self.adapter.get_filesystem()

    def open(self, relative_path: str, mode: str = "rb"):
        return self.fs().open(f"{self.config.fs_path.rstrip('/')}/{relative_path.lstrip('/')}", mode)

    def ls(self, path: str = ""):
        target = self.config.fs_path.rstrip("/")
        if path:
            target = f"{target}/{path.lstrip('/')}"
        return self.fs().ls(target)

    def invalidate(self) -> None:
        self.adapter.invalidate()


class LocalS3Profile(S3Catalog):
    """Backward-compatible alias for the legacy local S3 profile helper."""
