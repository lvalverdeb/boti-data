from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Union

from boti.core.models import ResourceConfig
from boti.core.security import is_valid_env_var_name
from boti.core.settings import SqlDatabaseSettings, load_prefixed_model
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from sqlalchemy.pool import NullPool, Pool, QueuePool, StaticPool

_ALLOWED_POOLCLASS_IMPORTS = {
    "sqlalchemy.pool.NullPool": NullPool,
    "sqlalchemy.pool.QueuePool": QueuePool,
    "sqlalchemy.pool.StaticPool": StaticPool,
}


class SqlDatabaseConfig(ResourceConfig):
    """Immutable configuration for SQLAlchemy engine initialization.

    ``query_only=True`` enforces read-only access at the database level (via
    a ``SET ... READ ONLY``-style statement issued on the SQLAlchemy
    ``"connect"`` event, not just by blocking ORM methods in Python) — this
    assumes ``connection_url`` points directly at Postgres/MySQL, or at a
    *session*-pooling-mode connection pooler (one physical backend connection
    dedicated to one logical client for its whole lifetime). It does **not**
    cover a *transaction*-pooling-mode pooler (e.g. PgBouncer in ``transaction``
    mode): there, a single physical connection can be handed to a different,
    unrelated logical client between transactions, and session-level state set
    once at physical-connect time is not guaranteed to survive that handoff.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection_url: SecretStr = Field(..., description="Database DSN")
    query_only: bool = Field(
        default=True,
        description=(
            "When true, native database-enforced read-only access is required. "
            "Set explicitly to false to allow writes. Assumes a direct connection "
            "or a session-pooling-mode proxy — transaction-pooling-mode poolers "
            "(e.g. PgBouncer 'transaction' mode) are out of scope; see class docstring."
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
        env_file: Union[str, Path] | None = None,
        **overrides: Any,
    ) -> SqlDatabaseConfig:
        return cls.from_env_prefix("DB_", env_file=env_file, **overrides)

    @classmethod
    def from_env_prefix(
        cls,
        prefix: str,
        *,
        env_file: Union[str, Path] | None = None,
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
            raise ValueError("worker_connection_env_var must be a valid environment variable name.")
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
                "worker_connection_env_var is not set; falling back to raw DSN. "
                "Credentials may be visible in scheduler logs.",
                UserWarning,
                stacklevel=2,
            )
            payload["connection_url"] = config.connection_url
        return cls(**payload)
