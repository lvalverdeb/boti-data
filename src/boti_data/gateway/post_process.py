"""Post-processing pipeline for DataGateway results.

Handles field-map translation, column renaming, fieldname filtering,
DataFrame options application, and diagnostics logging.
"""

from __future__ import annotations

import warnings
from time import perf_counter
from typing import Any, Literal

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
from boti.core.logger import Logger
from boti_dask import current_client_summary, describe_frame, safe_persist

from boti_data.db.partitioned_planner import SqlPartitionPlanner
from boti_data.field_map import FieldMap
from boti_data.schema import apply_schema_map

from .frame_strategies import FrameResult, FrameStrategy, get_frame_strategy
from .requests import DataFrameOptions, DataFrameParams, ResolvedReturnType, ReturnType

__all__ = [
    "PostProcessor",
    "frame_strategy",
    "strategy_for_frame",
]


def frame_strategy(return_type: ResolvedReturnType) -> FrameStrategy:
    return get_frame_strategy(return_type)


def strategy_for_frame(frame: FrameResult) -> FrameStrategy:
    if isinstance(frame, dd.DataFrame):
        return get_frame_strategy("dask")
    if isinstance(frame, pd.DataFrame):
        return get_frame_strategy("pandas")
    if isinstance(frame, pa.Table):
        return get_frame_strategy("arrow")
    if isinstance(frame, pl.DataFrame):
        return get_frame_strategy("polars")
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


class PostProcessor:
    """Post-processing pipeline for DataGateway results.

    Encapsulates result-shaping steps (field-map translation, column
    renaming, fieldname filtering, DataFrame options) and diagnostics
    logging so that DataGateway can focus on load orchestration.
    """

    def __init__(
        self,
        field_map: FieldMap,
        df_params: DataFrameParams,
        df_options: DataFrameOptions,
        *,
        backend: str | None = None,
        configured: bool = False,
        logger: Any | None = None,
    ) -> None:
        self._field_map = field_map
        self._df_params = df_params
        self._df_options = df_options
        self._backend = backend
        self._configured = configured
        self._logger = logger

    # ------------------------------------------------------------------
    # Logger lifecycle (updated when async resource becomes available)
    # ------------------------------------------------------------------

    def set_logger(self, logger: Any | None) -> None:
        self._logger = logger

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def finalize_load_result(
        self,
        df: Any,
        strategy: FrameStrategy,
        persist: bool,
        resilient: bool,
        dry_run: bool,
        diagnostics: bool,
        started: float,
    ) -> FrameResult:
        result = strategy.normalize(df)
        if persist and not dry_run and isinstance(result, dd.DataFrame):
            if resilient:
                result = safe_persist(result)
            else:
                result = get_frame_strategy("dask").persist(result)
        if diagnostics:
            self._log_load_complete(result, elapsed=perf_counter() - started)
        return result

    def finalize_configured_result(
        self,
        result: FrameResult,
        *,
        return_type: ReturnType,
        apply_field_map: bool,
        fieldnames: tuple[str, ...] | None = None,
    ) -> FrameResult:
        strategy = frame_strategy(return_type)
        frame = strategy.normalize(result)
        if apply_field_map:
            frame = self._apply_field_map(frame, strategy=strategy)
        return self._apply_df_options(
            self._apply_column_names(
                self.filter_to_fieldnames(frame, strategy=strategy, fieldnames=fieldnames),
                strategy=strategy,
            ),
            strategy=strategy,
        )

    def log_load_start(
        self,
        *,
        requested_return_type: ReturnType,
        resolved_return_type: ResolvedReturnType,
        requested_execution_mode: str | None,
        resolved_execution_mode: str | None,
        loader_return_type: Literal["pandas", "arrow", "dask"],
        persist: bool,
    ) -> None:
        logger = self._logger
        if logger is None:
            return
        if hasattr(logger, "set_level"):
            logger.set_level(Logger.INFO)
        logger.info(
            "Gateway load starting "
            f"backend={self._backend} configured={self._configured} "
            f"requested_return_type={requested_return_type} "
            f"resolved_return_type={resolved_return_type} "
            f"requested_execution_mode={requested_execution_mode} "
            f"resolved_execution_mode={resolved_execution_mode} "
            f"loader_return_type={loader_return_type} "
            f"persist={persist}"
        )
        client_summary = current_client_summary()
        if client_summary is not None:
            logger.info("Gateway load active Dask client=%s", client_summary)

    def coerce_eager_sql_frame(self, frame: pd.DataFrame, *, statement: Any) -> pd.DataFrame:
        meta_dtypes = SqlPartitionPlanner.infer_meta_dtypes(statement)
        return apply_schema_map(frame, meta_dtypes, require_columns=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_field_map(
        self,
        frame: FrameResult,
        *,
        strategy: FrameStrategy | None = None,
    ) -> FrameResult:
        strategy = strategy or strategy_for_frame(frame)
        if not self._field_map:
            return frame
        columns = (
            list(frame.column_names)
            if isinstance(frame, pa.Table)
            else list(frame.columns)
        )
        rename_map = {
            column: self._field_map.to_semantic(column)
            for column in columns
            if self._field_map.to_semantic(column) != column
        }
        return strategy.rename_columns(frame, rename_map)

    def _apply_column_names(
        self,
        frame: FrameResult,
        *,
        strategy: FrameStrategy | None = None,
    ) -> FrameResult:
        strategy = strategy or strategy_for_frame(frame)
        if self._df_params.column_names:
            frame = strategy.apply_column_names(frame, self._df_params.column_names)
        return frame

    def filter_to_fieldnames(
        self,
        frame: FrameResult,
        *,
        strategy: FrameStrategy | None = None,
        fieldnames: tuple[str, ...] | None = None,
    ) -> FrameResult:
        strategy = strategy or strategy_for_frame(frame)
        fieldnames = fieldnames if fieldnames is not None else self._df_params.fieldnames
        if not fieldnames:
            return frame

        present = set(frame.column_names if isinstance(frame, pa.Table) else frame.columns)
        missing = [f for f in fieldnames if f not in present]
        if missing:
            warnings.warn(
                f"DataGateway: requested fieldnames not found in result and will be skipped: {missing}",
                stacklevel=4,
            )
        valid = [f for f in fieldnames if f in present]
        if not valid:
            return frame
        return strategy.select_columns(frame, valid)

    def _apply_df_options(
        self,
        frame: FrameResult,
        *,
        strategy: FrameStrategy | None = None,
    ) -> FrameResult:
        strategy = strategy or strategy_for_frame(frame)
        return strategy.apply_options(frame, self._df_params, self._df_options)

    def _log_load_complete(self, frame: FrameResult, *, elapsed: float) -> None:
        logger = self._logger
        if logger is None:
            return
        if hasattr(logger, "set_level"):
            logger.set_level(Logger.INFO)
        logger.info(
            f"Gateway load completed elapsed={elapsed:.2f}s metrics={describe_frame(frame)}"
        )
        logger.info(
            f"Gateway load graph metrics={describe_frame(frame)}"
        )
