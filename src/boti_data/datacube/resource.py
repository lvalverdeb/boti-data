from __future__ import annotations

import asyncio
from typing import Any

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
from boti.core import Logger

from .contract import DatacubeConfig, DatacubeFrame


def _is_supported_frame(value: Any) -> bool:
    return isinstance(value, (pd.DataFrame, dd.DataFrame, pa.Table, pl.DataFrame))


class DatacubeResource:
    """Runtime wrapper that invokes datacube loader callables from config."""

    def __init__(self, config: DatacubeConfig) -> None:
        self.config = config
        self._is_closed = False
        self.logger = config.logger
        if self.logger is None:
            self.logger = Logger.default_logger(logger_name=self.__class__.__name__)

    @property
    def closed(self) -> bool:
        return self._is_closed

    def close(self) -> None:
        self._is_closed = True

    async def aclose(self) -> None:
        self._is_closed = True

    def _assert_open(self) -> None:
        if self._is_closed:
            raise RuntimeError("DatacubeResource is closed")

    def _validate_frame(self, value: Any) -> DatacubeFrame:
        if not _is_supported_frame(value):
            raise TypeError(
                "Datacube loader must return pandas, dask, pyarrow, or polars frame-like data. "
                f"Got {type(value)!r}."
            )
        return value

    def _with_defaults(self, request: Any) -> Any:
        if getattr(request, "cube", None) is not None:
            return request
        if self.config.default_cube is None:
            return request
        if hasattr(request, "model_copy"):
            return request.model_copy(update={"cube": self.config.default_cube})
        request.cube = self.config.default_cube
        return request

    def _prepare_request(self, request: Any) -> Any:
        resolved = self._with_defaults(request)
        contract = self.config.contract
        if contract is not None and contract.request_transformer is not None:
            resolved = contract.request_transformer(resolved)
        if contract is not None and contract.request_validator is not None:
            try:
                contract.request_validator(resolved)
            except Exception as exc:  # pragma: no cover - exercised through gateway tests
                raise ValueError(
                    "Datacube contract request validation failed: "
                    f"{exc}. "
                    f"{self._request_summary(resolved)} "
                    "Adjust DatacubeContract.request_validator or pass valid cube/filter inputs."
                ) from exc
        return resolved

    @staticmethod
    def _request_summary(request: Any) -> str:
        cube = getattr(request, "cube", None)
        filters = getattr(request, "filters", {})
        if isinstance(filters, dict):
            filter_keys = sorted(filters.keys())
        else:
            filter_keys = []
        return f"Request context cube={cube!r}, filter_keys={filter_keys}."

    def _transform_frame(self, frame: DatacubeFrame, request: Any) -> DatacubeFrame:
        contract = self.config.contract
        if contract is None or contract.frame_transformer is None:
            return frame
        return self._validate_frame(contract.frame_transformer(frame, request))

    def _resolve_sync_loader(self):
        contract = self.config.contract
        if contract is not None and contract.loader is not None:
            return contract.loader
        return self.config.loader

    def _resolve_async_loader(self):
        contract = self.config.contract
        if contract is not None and contract.async_loader is not None:
            return contract.async_loader
        return self.config.async_loader

    def load(self, request: Any) -> DatacubeFrame:
        self._assert_open()
        prepared_request = self._prepare_request(request)
        loader = self._resolve_sync_loader()
        if loader is None:
            raise RuntimeError("Datacube sync load requires DatacubeConfig.loader.")
        frame = self._validate_frame(loader(prepared_request))
        return self._transform_frame(frame, prepared_request)

    async def aload(self, request: Any) -> DatacubeFrame:
        self._assert_open()
        prepared_request = self._prepare_request(request)
        async_loader = self._resolve_async_loader()
        if async_loader is not None:
            frame = self._validate_frame(await async_loader(prepared_request))
            return self._transform_frame(frame, prepared_request)
        loader = self._resolve_sync_loader()
        if loader is None:
            raise RuntimeError("Datacube async load requires loader or async_loader.")
        frame = self._validate_frame(await asyncio.to_thread(loader, prepared_request))
        return self._transform_frame(frame, prepared_request)

