from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fsspec

from boti_data.datacube import DatacubeConfig
from boti_data.db import SqlDatabaseConfig
from boti_data.parquet.resource import ParquetDataConfig

from .requests import BackendConfig, DataFrameOptions, DataFrameParams


@dataclass(frozen=True)
class _GatewayCommonOptions:
    fs: fsspec.AbstractFileSystem | None
    fs_factory: Any | None
    table: str | None
    field_map: dict[str, str] | None
    sticky_filters: dict[str, Any] | None
    exclude: bool
    df_params: DataFrameParams | None
    df_options: DataFrameOptions | None

    def gateway_kwargs(self) -> dict[str, Any]:
        return {
            "fs": self.fs,
            "fs_factory": self.fs_factory,
            "table": self.table,
            "field_map": self.field_map,
            "sticky_filters": self.sticky_filters,
            "exclude": self.exclude,
            "df_params": self.df_params,
            "df_options": self.df_options,
        }


def coerce_backend_config(config: Any) -> BackendConfig:
    if isinstance(config, (SqlDatabaseConfig, ParquetDataConfig, DatacubeConfig)):
        return config
    if isinstance(config, tuple):
        if len(config) == 1 and isinstance(config[0], (SqlDatabaseConfig, ParquetDataConfig)):
            return config[0]
        raise TypeError(
            "Unsupported config type for DataGateway: tuple. "
            "Pass a SqlDatabaseConfig or ParquetDataConfig directly. "
            "If you created the config with a trailing comma, remove it."
        )
    raise TypeError(f"Unsupported config type for DataGateway: {type(config)!r}")


def extract_config_common_options(cfg: dict[str, Any]) -> tuple[str, _GatewayCommonOptions]:
    backend: str = cfg.pop("backend", "sqlalchemy")
    df_params_raw: dict[str, Any] | None = cfg.pop("df_params", None)
    df_options_raw: dict[str, Any] | None = cfg.pop("df_options", None)
    common = _GatewayCommonOptions(
        fs=cfg.pop("fs", None),
        fs_factory=cfg.pop("fs_factory", None),
        table=cfg.pop("table", None),
        field_map=cfg.pop("field_map", None),
        sticky_filters=cfg.pop("sticky_filters", None),
        exclude=bool(cfg.pop("exclude", cfg.pop("use_exclude", False))),
        df_params=DataFrameParams(**df_params_raw) if df_params_raw else None,
        df_options=DataFrameOptions(**df_options_raw) if df_options_raw else None,
    )
    return backend, common


def should_default_partition_on(cfg: dict[str, Any]) -> bool:
    return (
        "partition_on" not in cfg
        and cfg.get("parquet_start_date") is not None
        and cfg.get("parquet_end_date") is not None
    )
