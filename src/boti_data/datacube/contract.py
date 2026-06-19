from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
from boti.core import ResourceConfig
from pydantic import ConfigDict, Field, field_validator, model_validator

DatacubeFrame = pd.DataFrame | dd.DataFrame | pa.Table | pl.DataFrame
DatacubeSyncLoader = Callable[[Any], DatacubeFrame]
DatacubeAsyncLoader = Callable[[Any], Awaitable[DatacubeFrame]]
DatacubeRequestValidator = Callable[[Any], None]
DatacubeRequestTransformer = Callable[[Any], Any]
DatacubeFrameTransformer = Callable[[DatacubeFrame, Any], DatacubeFrame]


@dataclass(slots=True)
class DatacubeContract:
    """Optional typed datacube contract hooks used by DatacubeResource.

    Hooks are applied in this order:
    1) request_transformer
    2) request_validator
    3) loader/async_loader
    4) frame_transformer
    """

    loader: DatacubeSyncLoader | None = None
    async_loader: DatacubeAsyncLoader | None = None
    request_validator: DatacubeRequestValidator | None = None
    request_transformer: DatacubeRequestTransformer | None = None
    frame_transformer: DatacubeFrameTransformer | None = None


class DatacubeConfig(ResourceConfig):
    """Configuration for callable-backed datacube loading."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    loader: DatacubeSyncLoader | None = Field(default=None)
    async_loader: DatacubeAsyncLoader | None = Field(default=None)
    contract: DatacubeContract | None = Field(default=None)
    default_cube: str | None = Field(default=None)

    @field_validator("default_cube")
    @classmethod
    def _validate_default_cube(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("default_cube must not be empty.")
        return normalized

    @field_validator("contract")
    @classmethod
    def _validate_contract_type(cls, value: DatacubeContract | None) -> DatacubeContract | None:
        if value is None:
            return None
        if not isinstance(value, DatacubeContract):
            raise TypeError("contract must be an instance of DatacubeContract.")
        return value

    @model_validator(mode="after")
    def _validate_loader_presence(self) -> DatacubeConfig:
        contract_loader = self.contract.loader if self.contract is not None else None
        contract_async_loader = self.contract.async_loader if self.contract is not None else None
        if (
            self.loader is None
            and self.async_loader is None
            and contract_loader is None
            and contract_async_loader is None
        ):
            raise ValueError(
                "DatacubeConfig requires loader/async_loader or contract.loader/contract.async_loader."
            )
        return self

