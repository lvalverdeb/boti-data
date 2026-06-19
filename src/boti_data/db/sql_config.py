from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

from boti.core import ResourceConfig, SqlDatabaseSettings, is_valid_env_var_name, load_prefixed_model
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from sqlalchemy.pool import NullPool, Pool, QueuePool, StaticPool

_logger = logging.getLogger(__name__)
_ALLOWED_POOLCLASS_IMPORTS = {
    "sqlalchemy.pool.NullPool": NullPool,
    "sqlalchemy.pool.QueuePool": QueuePool,
    "sqlalchemy.pool.StaticPool": StaticPool,
}


class SqlDatabaseConfig(ResourceConfig):
    """Immutable configuration for SQLAlchemy engine initialization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection_url: SecretStr = Field(..., description="Database DSN")
    query_only: bool = Field(
        default=True,
        description=(
            "When true, native database-enforced read-only access is required. "
            "Set explicitly to false to allow writes."
        ),
    )
    worker_connection_env_var: str | None = Field(
        default=None,
        description=(
            "Optional environment variable name that worker processes use to resolve the DB DSN "
            "without serializing credentials inside distributed task payloads."
        ),
    )
    pool_size: int = Field(default=5, ge=0)
    max_overflow: int = Field(default=10, ge=0)
    pool_timeout: int = Field(default=30, ge=0)
    pool_recycle: int = Field(default=1800)
    pool_pre_ping: bool = Field(default=True)
    poolclass: type[Pool] = QueuePool
    connect_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Driver-specific connection arguments.",
    )
    execution_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine execution options",
    )

    @classmethod
    def from_settings(
        cls,
        settings: SqlDatabaseSettings,
        **overrides: Any,
    ) -> SqlDatabaseConfig:
        payload = settings.model_dump(exclude_none=True)
        payload.update(overrides)
        return cls(**payload)

    @classmethod
    def from_env(
        cls,
        *,
        env_file: str | Path | None = None,
        **overrides: Any,
    ) -> SqlDatabaseConfig:
        return cls.from_env_prefix("DB_", env_file=env_file, **overrides)

    @classmethod
    def from_env_prefix(
        cls,
        prefix: str,
        *,
        env_file: str | Path | None = None,
        **overrides: Any,
    ) -> SqlDatabaseConfig:
        settings = load_prefixed_model(SqlDatabaseSettings, prefix, env_file=env_file)
        return cls.from_settings(settings, **overrides)

    @field_validator("poolclass", mode="before")
    @classmethod
    def _resolve_poolclass(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value

        resolved = _ALLOWED_POOLCLASS_IMPORTS.get(value)
        if resolved is None:
            raise ValueError(
                "poolclass must be one of: sqlalchemy.pool.QueuePool, "
                "sqlalchemy.pool.NullPool, sqlalchemy.pool.StaticPool."
            )
        return resolved

    @field_validator("worker_connection_env_var")
    @classmethod
    def _validate_worker_connection_env_var(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not is_valid_env_var_name(value):
            raise ValueError(
                "worker_connection_env_var must be a valid environment variable name."
            )
        return value


class WorkerSqlConfig(BaseModel):
    """Minimal SQL configuration shipped to worker tasks."""

    model_config = ConfigDict(frozen=True)

    connection_url: SecretStr | None = Field(default=None)
    connection_env_var: str | None = Field(default=None)
    query_only: bool = Field(default=True)
    pool_recycle: int = Field(default=1800)
    pool_pre_ping: bool = Field(default=True)
    connect_args: dict[str, Any] = Field(default_factory=dict)
    execution_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("connection_env_var")
    @classmethod
    def _validate_connection_env_var(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not is_valid_env_var_name(value):
            raise ValueError("connection_env_var must be a valid environment variable name.")
        return value

    @model_validator(mode="after")
    def _validate_source(self) -> WorkerSqlConfig:
        if bool(self.connection_url) == bool(self.connection_env_var):
            raise ValueError(
                "WorkerSqlConfig requires exactly one of connection_url or connection_env_var."
            )
        return self

    @classmethod
    def from_database_config(cls, config: SqlDatabaseConfig) -> WorkerSqlConfig:
        payload: dict[str, Any] = {
            "query_only": config.query_only,
            "pool_recycle": config.pool_recycle,
            "pool_pre_ping": config.pool_pre_ping,
            "connect_args": config.connect_args,
            "execution_options": config.execution_options,
        }
        if config.worker_connection_env_var is not None:
            payload["connection_env_var"] = config.worker_connection_env_var
        else:
            warnings.warn(
                "Distributed SQL task payloads will carry the raw DSN credential because "
                "'worker_connection_env_var' is not set on SqlDatabaseConfig. "
                "Set 'worker_connection_env_var' to the name of an environment variable "
                "that resolves the DSN on each worker to avoid serializing credentials.",
                stacklevel=2,
                category=UserWarning,
            )
            _logger.warning(
                "WorkerSqlConfig created with raw DSN fallback; "
                "set worker_connection_env_var to avoid credential serialization."
            )
            payload["connection_url"] = config.connection_url
        return cls(**payload)
