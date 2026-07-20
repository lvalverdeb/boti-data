from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from boti_data.gateway.core import DataGateway
from boti_data.gateway.normalization import LOAD_CONTROL_KEYS
from boti_data.gateway.requests import BackendConfig, ExecutionMode, ReturnType
from boti_data.helper import DataHelper
from boti_data.parquet.reader_config import (
    UNSET,
    ParquetReaderConfigBuilder,
    ParquetReaderSettings,
)
from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource


class ParquetReader(DataHelper):
    """Parquet-specialized helper that preserves the gateway load/aload API."""

    def __init__(
        self,
        config: DataGateway | BackendConfig | Mapping[str, Any] | None = None,
        *,
        parquet_storage_path: str | None = None,
        storage_path: str | None = None,
        parquet_filename: str | None = None,
        parquet_start_date: str | None = None,
        parquet_end_date: str | None = None,
        partition_on: Sequence[str] | str | None | object = UNSET,
        filesystem_profile: str | None = None,
        parquet_max_age_minutes: int | None = None,
        return_type: ReturnType | None = None,
        execution_mode: ExecutionMode | None = None,
        fs: Any | None = None,
        fs_factory: Any | None = None,
        **overrides: Any,
    ) -> None:
        settings = ParquetReaderSettings(
            parquet_storage_path=parquet_storage_path,
            storage_path=storage_path,
            parquet_filename=parquet_filename,
            parquet_start_date=parquet_start_date,
            parquet_end_date=parquet_end_date,
            partition_on=partition_on,
            filesystem_profile=filesystem_profile,
            parquet_max_age_minutes=parquet_max_age_minutes,
            return_type=return_type,
            execution_mode=execution_mode,
        )
        if config is not None:
            ParquetReaderConfigBuilder.validate_no_explicit_settings(settings)
            if isinstance(config, Mapping) and "backend" not in config:
                config = ParquetReaderConfigBuilder.ensure_parquet_backend(config)
            super().__init__(config, fs=fs, fs_factory=fs_factory, **overrides)
        else:
            payload = ParquetReaderConfigBuilder.build_payload(settings, overrides)
            super().__init__(payload, fs=fs, fs_factory=fs_factory, **overrides)

        self._default_return_type = self.gateway.default_return_type
        self._default_execution_mode = self.gateway.default_execution_mode

    def _resolve_default_load_options(self, options: dict[str, Any]) -> dict[str, Any]:
        resolved_options = self._normalize_filter_options(options)
        if "return_type" not in resolved_options and "as_pandas" not in resolved_options:
            resolved_options["return_type"] = self._default_return_type
        if "execution_mode" not in resolved_options:
            resolved_options["execution_mode"] = self._default_execution_mode
        return resolved_options

    def load(self, **options: Any) -> Any:
        return super().load(**self._resolve_default_load_options(options))

    async def aload(self, **options: Any) -> Any:
        return await super().aload(**self._resolve_default_load_options(options))

    @staticmethod
    def _normalize_filter_options(options: Mapping[str, Any]) -> dict[str, Any]:
        resolved_options = dict(options)
        explicit_filters = resolved_options.get("filters")
        if explicit_filters is not None and not isinstance(explicit_filters, Mapping):
            raise TypeError("filters must be a mapping when provided.")

        bare_filters = {
            key: value for key, value in resolved_options.items() if key not in LOAD_CONTROL_KEYS
        }
        if not bare_filters:
            return resolved_options

        merged_filters = dict(explicit_filters or {})
        merged_filters.update(bare_filters)
        resolved_options["filters"] = merged_filters
        for key in bare_filters:
            resolved_options.pop(key, None)
        return resolved_options

    @property
    def config(self) -> ParquetDataConfig:
        config = self.gateway.config
        if not isinstance(config, ParquetDataConfig):
            raise TypeError(f"ParquetReader expected ParquetDataConfig, got {type(config)!r}")
        return config

    @property
    def resource(self) -> ParquetDataResource:
        resource = self.gateway.resource
        if not isinstance(resource, ParquetDataResource):
            raise TypeError(f"ParquetReader expected ParquetDataResource, got {type(resource)!r}")
        return resource

    @property
    def parquet_storage_path(self) -> str | None:
        return self.config.parquet_storage_path

    @property
    def parquet_filename(self) -> str | None:
        return self.config.parquet_filename

    @property
    def parquet_start_date(self) -> Any | None:
        return self.config.parquet_start_date

    @property
    def parquet_end_date(self) -> Any | None:
        return self.config.parquet_end_date

    @property
    def partition_on(self) -> list[str] | None:
        return self.config.partition_on

    @property
    def fs(self) -> Any | None:
        return self.resource.fs


__all__ = ["ParquetReader"]
