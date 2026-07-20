"""Validated configuration for Parquet-backed discovery and loading."""

from __future__ import annotations

import datetime as dt

from boti.core.models import ResourceConfig
from boti.core.security import is_valid_identifier
from pydantic import Field, field_validator, model_validator


class ParquetDataConfig(ResourceConfig):
    """Validated configuration for Parquet-backed discovery and loading."""

    parquet_storage_path: str | None = Field(default=None)
    parquet_filename: str | None = Field(default=None)
    parquet_start_date: dt.date | None = Field(default=None)
    parquet_end_date: dt.date | None = Field(default=None)
    parquet_max_age_minutes: int = Field(default=0, ge=0)
    partition_on: list[str] | None = Field(default=None)
    filesystem_profile: str | None = Field(default=None)

    @field_validator("parquet_storage_path")
    @classmethod
    def validate_storage_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("parquet_storage_path must be specified.")
        return normalized.rstrip("/")

    @field_validator("parquet_filename")
    @classmethod
    def validate_filename(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("parquet_filename must be a simple file name, not a path.")
        return value

    @field_validator("partition_on")
    @classmethod
    def validate_partition_on(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("partition_on must contain at least one partition field.")
        if any(not is_valid_identifier(item) for item in value):
            raise ValueError("partition_on values must be valid identifiers.")
        return value

    @field_validator("filesystem_profile")
    @classmethod
    def validate_filesystem_profile(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("filesystem_profile must not be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_date_range(self) -> ParquetDataConfig:
        if self.parquet_storage_path is None and self.filesystem_profile is None:
            raise ValueError("Either parquet_storage_path or filesystem_profile must be provided.")
        if bool(self.parquet_start_date) != bool(self.parquet_end_date):
            raise ValueError(
                "Both parquet_start_date and parquet_end_date must be provided, or neither."
            )
        if (
            self.parquet_start_date is not None
            and self.parquet_end_date is not None
            and self.parquet_end_date < self.parquet_start_date
        ):
            raise ValueError("parquet_end_date cannot be before parquet_start_date.")
        return self
