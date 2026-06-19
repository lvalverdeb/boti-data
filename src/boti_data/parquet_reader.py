from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from boti_data.gateway.core import DataGateway
from boti_data.gateway.normalization import LOAD_CONTROL_KEYS
from boti_data.gateway.requests import BackendConfig, ExecutionMode, ReturnType
from boti_data.helper import DataHelper
from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource

_UNSET = object()


class ParquetReader(DataHelper):
    """Parquet-specialized helper that preserves the gateway load/aload API."""

    DEFAULT_CONFIG: Final[dict[str, Any]] = {
        "backend": "parquet",
        "partition_on": ["partition_date"],
    }

    def __init__(
        self,
        config: DataGateway | BackendConfig | Mapping[str, Any] | None = None,
        *,
        parquet_storage_path: str | None = None,
        storage_path: str | None = None,
        parquet_filename: str | None = None,
        parquet_start_date: str | None = None,
        parquet_end_date: str | None = None,
        partition_on: Sequence[str] | str | None | object = _UNSET,
        filesystem_profile: str | None = None,
        parquet_max_age_minutes: int | None = None,
        return_type: ReturnType | None = None,
        execution_mode: ExecutionMode | None = None,
        fs: Any | None = None,
        fs_factory: Any | None = None,
        **overrides: Any,
    ) -> None:
        if config is not None:
            if any(
                value is not None
                for value in (
                    parquet_storage_path,
                    storage_path,
                    parquet_filename,
                    parquet_start_date,
                    parquet_end_date,
                    filesystem_profile,
                    parquet_max_age_minutes,
                    return_type,
                    execution_mode,
                )
            ) or partition_on is not _UNSET:
                raise TypeError(
                    "ParquetReader does not accept explicit parquet settings when config is provided."
                )
            super().__init__(config, fs=fs, fs_factory=fs_factory, **overrides)
            self._default_return_type = getattr(self.gateway, "_return_type", "dask")
            self._default_execution_mode = getattr(self.gateway, "_execution_mode", "auto")
            return

        if parquet_storage_path is not None and storage_path is not None:
            raise ValueError("Provide either parquet_storage_path or storage_path, not both.")

        payload: dict[str, Any] = dict(self.DEFAULT_CONFIG)
        resolved_path = parquet_storage_path or storage_path
        if resolved_path is not None:
            payload["parquet_storage_path"] = resolved_path
        if parquet_filename is not None:
            payload["parquet_filename"] = parquet_filename
        if parquet_start_date is not None:
            payload["parquet_start_date"] = parquet_start_date
        if parquet_end_date is not None:
            payload["parquet_end_date"] = parquet_end_date
        if filesystem_profile is not None:
            payload["filesystem_profile"] = filesystem_profile
        if parquet_max_age_minutes is not None:
            payload["parquet_max_age_minutes"] = parquet_max_age_minutes

        if partition_on is _UNSET:
            payload["partition_on"] = list(self.DEFAULT_CONFIG["partition_on"])
        elif partition_on is None:
            payload.pop("partition_on", None)
        elif isinstance(partition_on, str):
            payload["partition_on"] = [partition_on]
        else:
            payload["partition_on"] = list(partition_on)

        df_params_raw = overrides.pop("df_params", None)
        if df_params_raw is not None:
            if hasattr(df_params_raw, "model_dump"):
                df_params: dict[str, Any] = dict(df_params_raw.model_dump(exclude_none=True))
            elif isinstance(df_params_raw, Mapping):
                df_params = dict(df_params_raw)
            else:
                raise TypeError("df_params must be a mapping or DataFrameParams instance.")
        else:
            df_params = {}

        if return_type is not None:
            df_params["return_type"] = return_type
        if execution_mode is not None:
            df_params["execution_mode"] = execution_mode
        if df_params:
            payload["df_params"] = df_params

        super().__init__(payload, fs=fs, fs_factory=fs_factory, **overrides)
        self._default_return_type = getattr(self.gateway, "_return_type", "dask")
        self._default_execution_mode = getattr(self.gateway, "_execution_mode", "auto")

    def load(self, **options: Any) -> Any:
        resolved_options = self._normalize_filter_options(options)
        if "return_type" not in resolved_options and "as_pandas" not in resolved_options:
            resolved_options["return_type"] = self._default_return_type
        if "execution_mode" not in resolved_options:
            resolved_options["execution_mode"] = self._default_execution_mode
        return super().load(**resolved_options)

    async def aload(self, **options: Any) -> Any:
        resolved_options = self._normalize_filter_options(options)
        if "return_type" not in resolved_options and "as_pandas" not in resolved_options:
            resolved_options["return_type"] = self._default_return_type
        if "execution_mode" not in resolved_options:
            resolved_options["execution_mode"] = self._default_execution_mode
        return await super().aload(**resolved_options)

    @staticmethod
    def _normalize_filter_options(options: Mapping[str, Any]) -> dict[str, Any]:
        resolved_options = dict(options)
        explicit_filters = resolved_options.get("filters")
        if explicit_filters is not None and not isinstance(explicit_filters, Mapping):
            raise TypeError("filters must be a mapping when provided.")

        bare_filters = {
            key: value
            for key, value in resolved_options.items()
            if key not in LOAD_CONTROL_KEYS
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
