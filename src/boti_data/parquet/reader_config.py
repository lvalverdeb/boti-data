from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from boti_data.gateway.requests import ExecutionMode, ReturnType

UNSET: Final = object()


@dataclass(frozen=True)
class ParquetReaderSettings:
    parquet_storage_path: str | None = None
    storage_path: str | None = None
    parquet_filename: str | None = None
    parquet_start_date: str | None = None
    parquet_end_date: str | None = None
    partition_on: Sequence[str] | str | None | object = UNSET
    filesystem_profile: str | None = None
    parquet_max_age_minutes: int | None = None
    return_type: ReturnType | None = None
    execution_mode: ExecutionMode | None = None


class ParquetReaderConfigBuilder:
    """Builds legacy gateway config payloads from ParquetReader kwargs."""

    DEFAULT_CONFIG: Final[dict[str, Any]] = {
        "backend": "parquet",
        "partition_on": ["partition_date"],
    }

    _PARQUET_FIELDS: Final[tuple[str, ...]] = (
        "parquet_filename",
        "parquet_start_date",
        "parquet_end_date",
        "filesystem_profile",
        "parquet_max_age_minutes",
    )

    @classmethod
    def ensure_parquet_backend(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        if "backend" in config:
            return dict(config)
        return {**config, "backend": "parquet"}

    @classmethod
    def validate_no_explicit_settings(cls, settings: ParquetReaderSettings) -> None:
        explicit_values = (
            settings.parquet_storage_path,
            settings.storage_path,
            settings.parquet_filename,
            settings.parquet_start_date,
            settings.parquet_end_date,
            settings.filesystem_profile,
            settings.parquet_max_age_minutes,
            settings.return_type,
            settings.execution_mode,
        )
        if (
            any(value is not None for value in explicit_values)
            or settings.partition_on is not UNSET
        ):
            raise TypeError(
                "ParquetReader does not accept explicit parquet settings when config is provided."
            )

    @classmethod
    def build_payload(
        cls,
        settings: ParquetReaderSettings,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        if settings.parquet_storage_path is not None and settings.storage_path is not None:
            raise ValueError("Provide either parquet_storage_path or storage_path, not both.")

        payload: dict[str, Any] = dict(cls.DEFAULT_CONFIG)
        resolved_path = settings.parquet_storage_path or settings.storage_path
        if resolved_path is not None:
            payload["parquet_storage_path"] = resolved_path

        for key in cls._PARQUET_FIELDS:
            value = getattr(settings, key)
            if value is not None:
                payload[key] = value

        payload["partition_on"] = cls.resolve_partition_on(settings.partition_on)
        df_params = cls.resolve_df_params(
            overrides.pop("df_params", None),
            settings.return_type,
            settings.execution_mode,
        )
        if df_params:
            payload["df_params"] = df_params

        return payload

    @classmethod
    def resolve_partition_on(cls, partition_on: Any) -> list[str] | None:
        if partition_on is UNSET:
            return list(cls.DEFAULT_CONFIG["partition_on"])
        if partition_on is None:
            return None
        return [partition_on] if isinstance(partition_on, str) else list(partition_on)

    @staticmethod
    def resolve_df_params(
        raw_params: Any,
        return_type: ReturnType | None,
        execution_mode: ExecutionMode | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if raw_params is not None:
            if hasattr(raw_params, "model_dump"):
                params = dict(raw_params.model_dump(exclude_none=True))
            elif isinstance(raw_params, Mapping):
                params = dict(raw_params)
            else:
                raise TypeError("df_params must be a mapping or DataFrameParams instance.")

        if return_type is not None:
            params["return_type"] = return_type
        if execution_mode is not None:
            params["execution_mode"] = execution_mode
        return params
