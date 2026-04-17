"""
Dask-first gateway over existing boti_data backend resources.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any, Literal

import dask
import dask.dataframe as dd
import fsspec
import pandas as pd
import polars as pl
import pyarrow as pa
from boti.core.logger import Logger
from pydantic import SecretStr
from sqlalchemy.engine import url as sqlalchemy_url
from sqlalchemy.exc import SQLAlchemyError

from boti_data.db import (
    AsyncSqlDatabaseResource,
    SqlDatabaseConfig,
    SqlDatabaseResource,
    SqlPartitionedLoadRequest,
)
from boti_data.db.partitioned_execution import SqlPartitionExecutor
from boti_data.db.partitioned_planner import SqlPartitionPlanner
from boti_data.db.sql_config import WorkerSqlConfig
from boti_data.db.sql_engine import _get_worker_engine_identity
from boti_dask import (
    async_safe_head,
    current_client_summary,
    describe_frame,
    inspect_graph,
    safe_head,
    safe_persist,
)
from boti_data.field_map import FieldMap
from boti_data.filters import FilterHandler
from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource
from boti_data.schema import apply_schema_map

from .frame_strategies import FrameResult, FrameStrategy, get_frame_strategy
from .loaders import (
    build_backend_resource,
    build_sql_partitioned_request,
    load_parquet,
    load_sql,
    load_sql_partitioned,
    read_sql_async,
    reflect_and_select,
    reflect_and_select_async,
)
from .normalization import (
    DEFAULT_IN_CHUNK_SIZE as _DEFAULT_IN_CHUNK_SIZE,
)
from .normalization import (
    LOAD_CONTROL_KEYS as _LOAD_CONTROL_KEYS,
)
from .normalization import (
    build_partitioned_load_options,
    normalize_configured_filters,
    prepare_period_filters,
    split_control_and_filters,
)
from .requests import (
    BackendConfig,
    BackendName,
    BackendResource,
    DataFrameOptions,
    DataFrameParams,
    ExecutionMode,
    ParquetLoadRequest,
    ResolvedExecutionMode,
    ResolvedReturnType,
    ReturnType,
    SqlLoadRequest,
)

_AUTO_EAGER_MAX_ROWS = 10_000
_AUTO_EAGER_MAX_FILES = 4
_AUTO_EAGER_MAX_BYTES = 32 * 1024 * 1024


def _coerce_backend_config(config: Any) -> BackendConfig:
    if isinstance(config, (SqlDatabaseConfig, ParquetDataConfig)):
        return config
    if isinstance(config, (SqlDatabaseResource, AsyncSqlDatabaseResource)):
        return config.config
    if isinstance(config, tuple):
        if len(config) == 1 and isinstance(config[0], (SqlDatabaseConfig, ParquetDataConfig)):
            return config[0]
        raise TypeError(
            "Unsupported config type for DataGateway: tuple. "
            "Pass a SqlDatabaseConfig or ParquetDataConfig directly. "
            "If you created the config with a trailing comma, remove it."
        )
    raise TypeError(f"Unsupported config type for DataGateway: {type(config)!r}")


class DataGateway:
    """Dask-first gateway that delegates to existing backend resources.

    There are two usage modes:

    **Structured mode** (default)
        Pass explicit ``statement``, ``model``, ``filters``, etc. to
        :meth:`load` / :meth:`aload` exactly as before.

    **Configured mode**
        Set ``table=`` at construction time.  All keyword arguments passed
        to :meth:`load` / :meth:`aload` that are **not** control keywords are
        treated as runtime filter values and merged with ``sticky_filters``.

        All filter keys and ``fieldnames`` are always expressed as **semantic
        names** (the right-hand side of *field_map*).  When a ``field_map`` is
        provided the gateway translates them to DB column names before the query
        and renames results back to semantic names on return.  When no
        ``field_map`` is given the DB is assumed to already use semantic names
        and no translation is performed.

        Pass ``exclude=True`` to invert the combined filter — the query returns
        all rows that do **not** match the specified conditions (``WHERE NOT
        (condition1 AND condition2 AND ...)``).

        Example::

            gw = DataGateway(
                SqlDatabaseConfig(connection_url=SecretStr(db_url)),
                table="asm_tracking_productos",
                field_map=product_fields,      # present → DB has legacy names
                sticky_filters={"product_type_id": 1},
                exclude=True,                  # return rows where type != 1
            )
            df = await gw.aload(global_track_id=1)   # returns Dask DataFrame
            df = df.compute()
    """

    def __init__(
        self,
        config: BackendConfig,
        *,
        field_map: dict[str, str] | None = None,
        table: str | None = None,
        sticky_filters: dict[str, Any] | None = None,
        exclude: bool = False,
        df_params: DataFrameParams | None = None,
        df_options: DataFrameOptions | None = None,
        fs: fsspec.AbstractFileSystem | None = None,
        fs_factory: Any | None = None,
    ) -> None:
        config = _coerce_backend_config(config)
        self.config = config

        # For async DSNs the sync SqlDatabaseResource cannot be created eagerly;
        # defer to the AsyncSqlDatabaseResource path used by aload/__aenter__.
        if isinstance(config, SqlDatabaseConfig):
            _parsed = sqlalchemy_url.make_url(config.connection_url.get_secret_value())
            if _parsed.get_dialect().is_async:
                self.backend: BackendName = "sqlalchemy"
                self.resource: BackendResource | None = None
            else:
                self.backend, self.resource = build_backend_resource(
                    config, fs=fs, fs_factory=fs_factory
                )
        else:
            self.backend, self.resource = build_backend_resource(
                config, fs=fs, fs_factory=fs_factory
            )

        self._async_sql_resource: AsyncSqlDatabaseResource | None = None

        # Configured-mode state
        self._table = table
        self._field_map = FieldMap(field_map) if field_map else FieldMap({})
        self._sticky_filters: dict[str, Any] = dict(sticky_filters or {})
        self._exclude = exclude
        self._df_params = df_params or DataFrameParams()
        self._df_options = df_options or DataFrameOptions()
        self._return_type: ReturnType = self._df_params.return_type
        self._execution_mode: ExecutionMode = self._df_params.execution_mode
        self._configured_select_cache: dict[tuple[str, ...], tuple[Any, Any]] = {}
        self._configured_async_select_cache: dict[tuple[str, ...], tuple[Any, Any]] = {}

    @property
    def _configured(self) -> bool:
        """True when the gateway was set up with a fixed table name."""
        return self._table is not None

    @property
    def _logger(self) -> Any | None:
        if self._async_sql_resource is not None:
            return getattr(self._async_sql_resource, "logger", None)
        if self.resource is not None:
            return getattr(self.resource, "logger", None)
        return None

    @staticmethod
    def _frame_strategy(return_type: ResolvedReturnType) -> FrameStrategy:
        return get_frame_strategy(return_type)

    @staticmethod
    def _strategy_for_frame(frame: FrameResult) -> FrameStrategy:
        if isinstance(frame, dd.DataFrame):
            return get_frame_strategy("dask")
        if isinstance(frame, pd.DataFrame):
            return get_frame_strategy("pandas")
        if isinstance(frame, pa.Table):
            return get_frame_strategy("arrow")
        if isinstance(frame, pl.DataFrame):
            return get_frame_strategy("polars")
        raise TypeError(f"Unsupported frame type: {type(frame)!r}")

    @staticmethod
    def _resolved_execution_mode_for_return_type(
        return_type: ResolvedReturnType,
    ) -> ResolvedExecutionMode:
        return "lazy" if return_type == "dask" else "eager"

    @staticmethod
    def _loader_return_type(
        return_type: ResolvedReturnType,
        *,
        backend: BackendName,
        execution_mode: ResolvedExecutionMode,
    ) -> Literal["pandas", "arrow", "dask"]:
        if execution_mode == "lazy":
            return "dask"
        return DataGateway._frame_strategy(return_type).loader_return_type

    @staticmethod
    def _loader_as_pandas(
        *,
        backend: BackendName,
        execution_mode: ResolvedExecutionMode,
        loader_return_type: Literal["pandas", "arrow", "dask"],
    ) -> bool:
        return loader_return_type == "pandas"

    @staticmethod
    def _is_small_sql_result(total_rows: int, *, limit: int | None = None) -> bool:
        effective_rows = min(total_rows, limit) if limit is not None else total_rows
        return effective_rows <= _AUTO_EAGER_MAX_ROWS

    def _resolve_requested_return_type(self, *, as_pandas: bool, options: dict[str, Any]) -> ReturnType:
        if "return_type" in options:
            return options.pop("return_type")
        if as_pandas:
            return "pandas"
        if self._configured:
            return self._return_type
        return "dask"

    def _resolve_requested_execution_mode(self, options: dict[str, Any]) -> ExecutionMode:
        if "execution_mode" in options:
            return options.pop("execution_mode")
        if options.get("partitioned") is True:
            return "lazy"
        if options.get("partitioned") is False:
            return "eager"
        if self._configured:
            return self._execution_mode
        return "auto"

    @staticmethod
    def _auto_sql_threshold(limit: int | None = None) -> int:
        effective_limit = _AUTO_EAGER_MAX_ROWS if limit is None else min(limit, _AUTO_EAGER_MAX_ROWS)
        return max(0, effective_limit)

    def _resolve_execution_plan(
        self,
        *,
        requested_return_type: ReturnType,
        requested_execution_mode: ExecutionMode,
        options: dict[str, Any],
    ) -> tuple[ResolvedReturnType, ResolvedExecutionMode]:
        if requested_return_type == "auto":
            if requested_execution_mode == "auto":
                resolved_return_type = self._resolve_auto_return_type(options)
            else:
                resolved_return_type = "dask" if requested_execution_mode == "lazy" else "pandas"
        else:
            resolved_return_type = requested_return_type
        if requested_execution_mode == "auto":
            resolved_execution_mode = self._resolved_execution_mode_for_return_type(
                resolved_return_type
            )
        else:
            resolved_execution_mode = requested_execution_mode
        return resolved_return_type, resolved_execution_mode

    async def _resolve_execution_plan_async(
        self,
        *,
        requested_return_type: ReturnType,
        requested_execution_mode: ExecutionMode,
        options: dict[str, Any],
    ) -> tuple[ResolvedReturnType, ResolvedExecutionMode]:
        if requested_return_type == "auto":
            if requested_execution_mode == "auto":
                resolved_return_type = await self._resolve_auto_return_type_async(options)
            else:
                resolved_return_type = "dask" if requested_execution_mode == "lazy" else "pandas"
        else:
            resolved_return_type = requested_return_type
        if requested_execution_mode == "auto":
            resolved_execution_mode = self._resolved_execution_mode_for_return_type(
                resolved_return_type
            )
        else:
            resolved_execution_mode = requested_execution_mode
        return resolved_return_type, resolved_execution_mode

    @staticmethod
    def _configured_select_cache_key(db_columns: list[str] | None) -> tuple[str, ...]:
        return tuple(db_columns or ())

    def _get_configured_select(
        self,
        db_columns: list[str] | None,
    ) -> tuple[Any, Any]:
        cache_key = self._configured_select_cache_key(db_columns)
        cached = self._configured_select_cache.get(cache_key)
        if cached is not None:
            return cached
        assert isinstance(self.resource, SqlDatabaseResource)
        result = reflect_and_select(self.resource, self._table, db_columns)  # type: ignore[arg-type]
        self._configured_select_cache[cache_key] = result
        return result

    async def _get_configured_select_async(
        self,
        resource: Any,
        db_columns: list[str] | None,
    ) -> tuple[Any, Any]:
        cache_key = self._configured_select_cache_key(db_columns)
        cached = self._configured_async_select_cache.get(cache_key)
        if cached is not None:
            return cached
        result = await reflect_and_select_async(resource, self._table, db_columns)
        self._configured_async_select_cache[cache_key] = result
        return result

    def _resolve_auto_sql_return_type(
        self,
        options: dict[str, Any],
    ) -> ResolvedReturnType:
        if options.get("partitioned") is True:
            return "dask"
        if options.get("partitioned") is False:
            return "pandas"
        limit = options.get("limit")
        if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
            return "pandas"
        if options.get("sql") is not None:
            return "pandas"
        if options.get("statement") is None or options.get("model") is None:
            return "pandas"
        assert isinstance(self.resource, SqlDatabaseResource)
        request = build_sql_partitioned_request(options)
        threshold = self._auto_sql_threshold(request.limit)
        if threshold == 0:
            return "pandas"
        planner = SqlPartitionPlanner(self.resource)
        statement = planner.prepare_statement(request)
        probed_rows = planner.count_rows_up_to(statement, threshold)
        return "pandas" if probed_rows <= threshold else "dask"

    def _resolve_auto_sql_return_type_configured(
        self,
        options: dict[str, Any],
    ) -> ResolvedReturnType:
        model, stmt, db_filters, control = self._build_configured_request(options)
        if control.get("partitioned") is True:
            return "dask"
        if control.get("partitioned") is False:
            return "pandas"
        limit = control.get("limit")
        if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
            return "pandas"
        partitioned_options = build_partitioned_load_options(
            statement=stmt,
            model=model,
            filters=db_filters,
            control=control,
            default_chunk_size=self._df_params.chunk_size,
        )
        request = build_sql_partitioned_request(partitioned_options)
        assert isinstance(self.resource, SqlDatabaseResource)
        threshold = self._auto_sql_threshold(request.limit)
        if threshold == 0:
            return "pandas"
        planner = SqlPartitionPlanner(self.resource)
        statement = planner.prepare_statement(request)
        probed_rows = planner.count_rows_up_to(statement, threshold)
        return "pandas" if probed_rows <= threshold else "dask"

    async def _resolve_auto_sql_return_type_async(
        self,
        options: dict[str, Any],
    ) -> ResolvedReturnType:
        if options.get("partitioned") is True:
            return "dask"
        if options.get("partitioned") is False:
            return "pandas"
        limit = options.get("limit")
        if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
            return "pandas"
        if options.get("sql") is not None:
            return "pandas"
        if options.get("statement") is None or options.get("model") is None:
            return "pandas"
        request = build_sql_partitioned_request(options)
        threshold = self._auto_sql_threshold(request.limit)
        if threshold == 0:
            return "pandas"
        if self.resource is not None:
            assert isinstance(self.resource, SqlDatabaseResource)
            planner = SqlPartitionPlanner(self.resource)
            statement = planner.prepare_statement(request)
            probed_rows = await asyncio.to_thread(planner.count_rows_up_to, statement, threshold)
            return "pandas" if probed_rows <= threshold else "dask"

        async_resource = self._async_sql_resource
        if async_resource is None:
            assert isinstance(self.config, SqlDatabaseConfig)
            async with AsyncSqlDatabaseResource(self.config) as temp_resource:
                class _EngineAdapter:
                    engine = temp_resource.engine.sync_engine
                    logger = temp_resource.logger
                    debug = False

                planner = SqlPartitionPlanner(_EngineAdapter())  # type: ignore[arg-type]
                statement = planner.prepare_statement(request)
                async with temp_resource.engine.connect() as conn:
                    probed_rows = await planner._async_count_rows_up_to(statement, threshold, conn)
        else:
            class _EngineAdapter:
                engine = async_resource.engine.sync_engine
                logger = async_resource.logger
                debug = False

            planner = SqlPartitionPlanner(_EngineAdapter())  # type: ignore[arg-type]
            statement = planner.prepare_statement(request)
            async with async_resource.engine.connect() as conn:
                probed_rows = await planner._async_count_rows_up_to(statement, threshold, conn)
        return "pandas" if probed_rows <= threshold else "dask"

    async def _resolve_auto_sql_return_type_configured_async(
        self,
        options: dict[str, Any],
    ) -> ResolvedReturnType:
        if self.resource is not None:
            return await asyncio.to_thread(self._resolve_auto_sql_return_type_configured, options)

        assert isinstance(self.config, SqlDatabaseConfig)
        normalized = normalize_configured_filters(
            options,
            sticky_filters=self._sticky_filters,
            exclude=self._exclude,
        )
        control = normalized.control
        if control.get("partitioned") is True:
            return "dask"
        if control.get("partitioned") is False:
            return "pandas"
        limit = control.get("limit")
        if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
            return "pandas"
        combined_filters = normalized.filters

        if self._field_map:
            db_filters = self._field_map.translate_filters_to_db(
                combined_filters, input_keys_are="semantic"
            )
            db_columns: list[str] | None = (
                self._field_map.select_db_columns(self._df_params.fieldnames)
                if self._df_params.fieldnames
                else None
            )
        else:
            db_filters = combined_filters
            db_columns = list(self._df_params.fieldnames) if self._df_params.fieldnames else None

        async_resource = self._async_sql_resource
        if async_resource is None:
            async with AsyncSqlDatabaseResource(self.config) as temp_resource:
                model, stmt = await self._get_configured_select_async(temp_resource, db_columns)

                class _EngineAdapter:
                    engine = temp_resource.engine.sync_engine
                    logger = temp_resource.logger
                    debug = False

                planner = SqlPartitionPlanner(_EngineAdapter())  # type: ignore[arg-type]
                partitioned_options = build_partitioned_load_options(
                    statement=stmt,
                    model=model,
                    filters=db_filters,
                    control=control,
                    default_chunk_size=self._df_params.chunk_size,
                )
                request = build_sql_partitioned_request(partitioned_options)
                threshold = self._auto_sql_threshold(request.limit)
                if threshold == 0:
                    return "pandas"
                statement = planner.prepare_statement(request)
                async with temp_resource.engine.connect() as conn:
                    probed_rows = await planner._async_count_rows_up_to(statement, threshold, conn)
        else:
            model, stmt = await self._get_configured_select_async(async_resource, db_columns)

            class _EngineAdapter:
                engine = async_resource.engine.sync_engine
                logger = async_resource.logger
                debug = False

            planner = SqlPartitionPlanner(_EngineAdapter())  # type: ignore[arg-type]
            partitioned_options = build_partitioned_load_options(
                statement=stmt,
                model=model,
                filters=db_filters,
                control=control,
                default_chunk_size=self._df_params.chunk_size,
            )
            request = build_sql_partitioned_request(partitioned_options)
            threshold = self._auto_sql_threshold(request.limit)
            if threshold == 0:
                return "pandas"
            statement = planner.prepare_statement(request)
            async with async_resource.engine.connect() as conn:
                probed_rows = await planner._async_count_rows_up_to(statement, threshold, conn)
        return "pandas" if probed_rows <= threshold else "dask"

    def _estimate_parquet_bytes(self, resource: ParquetDataResource) -> int | None:
        try:
            files_to_load = resource._resolve_files_to_load()
        except Exception:
            return None
        if len(files_to_load) > _AUTO_EAGER_MAX_FILES:
            return None
        total_bytes = 0
        for path in files_to_load:
            info = resource._get_file_info(path)
            if info is None:
                return None
            size = info.get("size")
            if not isinstance(size, int):
                return None
            total_bytes += size
        return total_bytes

    def _resolve_auto_parquet_return_type(
        self,
        options: dict[str, Any],
    ) -> ResolvedReturnType:
        limit = options.get("limit")
        if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
            return "pandas"
        assert isinstance(self.resource, ParquetDataResource)
        try:
            files_to_load, total_bytes = self._resolve_parquet_scan_summary(self.resource)
        except Exception:
            return "dask"
        if not files_to_load:
            return "pandas"
        if len(files_to_load) > _AUTO_EAGER_MAX_FILES:
            return "dask"
        if total_bytes is None:
            return "dask"
        return "pandas" if total_bytes <= _AUTO_EAGER_MAX_BYTES else "dask"

    def _resolve_parquet_scan_summary(
        self,
        resource: ParquetDataResource,
    ) -> tuple[list[str], int | None]:
        files_to_load = resource._resolve_files_to_load()
        if not files_to_load:
            return [], 0
        if len(files_to_load) > _AUTO_EAGER_MAX_FILES:
            return files_to_load, None
        total_bytes = 0
        for path in files_to_load:
            info = resource._get_file_info(path)
            if info is None:
                return files_to_load, None
            size = info.get("size")
            if not isinstance(size, int):
                return files_to_load, None
            total_bytes += size
        return files_to_load, total_bytes

    def _prepare_structured_loader_options(self, options: dict[str, Any]) -> dict[str, Any]:
        if self._configured or self.backend != "sqlalchemy":
            return options
        columns = options.get("columns")
        statement = options.get("statement")
        model = options.get("model")
        if not columns or statement is None or model is None:
            return options
        projected_names = [
            self._field_map.to_db(column) if self._field_map else column
            for column in columns
        ]
        projected_columns = [getattr(model, column) for column in projected_names]
        prepared = dict(options)
        prepared["statement"] = statement.with_only_columns(
            *projected_columns,
            maintain_column_froms=True,
        )
        prepared.pop("columns", None)
        return prepared

    @staticmethod
    def _structured_sql_request_payload(
        options: dict[str, Any],
        *,
        return_type: ResolvedReturnType,
    ) -> dict[str, Any]:
        return {
            "sql": options.get("sql"),
            "statement": options.get("statement"),
            "model": options.get("model"),
            "filters": options.get("filters", {}),
            "params": options.get("params", {}),
            "limit": options.get("limit"),
            "columns": options.get("columns"),
            "as_pandas": bool(options.get("as_pandas", False)),
            "diagnostics": bool(options.get("diagnostics", False)),
            "return_type": return_type,
        }

    @staticmethod
    def _structured_parquet_request_payload(
        options: dict[str, Any],
        *,
        return_type: ResolvedReturnType,
    ) -> dict[str, Any]:
        return {
            "filters": options.get("filters", {}),
            "raw_filters": options.get("raw_filters"),
            "limit": options.get("limit"),
            "columns": options.get("columns"),
            "as_pandas": bool(options.get("as_pandas", False)),
            "diagnostics": bool(options.get("diagnostics", False)),
            "return_type": return_type,
        }

    def _resolve_auto_return_type(
        self,
        options: dict[str, Any],
    ) -> ResolvedReturnType:
        if self.backend == "sqlalchemy":
            if self._configured:
                return self._resolve_auto_sql_return_type_configured(options)
            return self._resolve_auto_sql_return_type(options)
        if self.backend == "parquet":
            return self._resolve_auto_parquet_return_type(options)
        return "dask"

    async def _resolve_auto_return_type_async(
        self,
        options: dict[str, Any],
    ) -> ResolvedReturnType:
        if self.backend == "sqlalchemy":
            if self._configured:
                return await self._resolve_auto_sql_return_type_configured_async(options)
            return await self._resolve_auto_sql_return_type_async(options)
        if self.backend == "parquet":
            return self._resolve_auto_parquet_return_type(options)
        return "dask"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_backend(
        cls,
        backend: BackendName,
        *,
        fs: fsspec.AbstractFileSystem | None = None,
        fs_factory: Any | None = None,
        **config_kwargs: Any,
    ) -> DataGateway:
        if backend == "sqlalchemy":
            return cls(SqlDatabaseConfig(**config_kwargs), fs=fs, fs_factory=fs_factory)
        if backend == "parquet":
            return cls(ParquetDataConfig(**config_kwargs), fs=fs, fs_factory=fs_factory)
        raise ValueError(f"Unsupported backend: {backend!r}")

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        **overrides: Any,
    ) -> DataGateway:
        """Build a :class:`DataGateway` from a legacy-style config dict.

        Accepts the same keys that ``DfHelper`` used to accept::

            DataGateway.from_config({
                "backend": "sqlalchemy",
                "connection_url": db_url,
                "table": "asm_tracking_productos",
                "field_map": product_fields,
                "sticky_filters": {"product_type_id": 1},
                "df_params": {
                    "fieldnames": ("id", "process_track_id"),
                    "column_names": ["id", "process_track_id_x"],
                    "chunk_size": 50000,
                    "index_col": "id",
                },
                "df_options": {
                    "sort_field": "global_track_id",
                    "duplicate_expr": ["id"],
                    "duplicate_keep": "last",
                },
            })

        Translation is automatically enabled whenever a ``field_map`` is
        provided.
        """
        cfg = dict(config)
        cfg.update(overrides)

        backend: str = cfg.pop("backend", "sqlalchemy")
        fs = cfg.pop("fs", None)
        fs_factory = cfg.pop("fs_factory", None)
        raw_url = cfg.pop("connection_url", None)
        table: str | None = cfg.pop("table", None)
        field_map: dict[str, str] | None = cfg.pop("field_map", None)
        sticky_filters: dict[str, Any] | None = cfg.pop("sticky_filters", None)
        # accept both 'exclude' and the legacy 'use_exclude' key
        exclude: bool = bool(cfg.pop("exclude", cfg.pop("use_exclude", False)))
        df_params_raw: dict[str, Any] | None = cfg.pop("df_params", None)
        df_params = DataFrameParams(**df_params_raw) if df_params_raw else None
        df_options_raw: dict[str, Any] | None = cfg.pop("df_options", None)
        df_options = DataFrameOptions(**df_options_raw) if df_options_raw else None

        if backend == "sqlalchemy":
            if raw_url is None:
                raise ValueError("from_config requires 'connection_url'.")
            connection_url = (
                SecretStr(raw_url) if isinstance(raw_url, str) else raw_url
            )
            db_config = SqlDatabaseConfig(connection_url=connection_url, **cfg)
            return cls(
                db_config,
                fs=fs,
                fs_factory=fs_factory,
                table=table,
                field_map=field_map,
                sticky_filters=sticky_filters,
                exclude=exclude,
                df_params=df_params,
                df_options=df_options,
            )
        if backend == "parquet":
            storage_path = cfg.pop("storage_path", None)
            if storage_path is not None and "parquet_storage_path" not in cfg:
                cfg["parquet_storage_path"] = storage_path
            if (
                "partition_on" not in cfg
                and cfg.get("parquet_start_date") is not None
                and cfg.get("parquet_end_date") is not None
            ):
                cfg["partition_on"] = ["partition_date"]
            parquet_config = ParquetDataConfig(**cfg)
            return cls(
                parquet_config,
                fs=fs,
                fs_factory=fs_factory,
                table=table,
                field_map=field_map,
                sticky_filters=sticky_filters,
                exclude=exclude,
                df_params=df_params,
                df_options=df_options,
            )
        raise ValueError(f"Unsupported backend: {backend!r}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self.resource is not None:
            self.resource.close()

    def __enter__(self) -> DataGateway:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> DataGateway:
        if self.backend == "sqlalchemy":
            assert isinstance(self.config, SqlDatabaseConfig)
            parsed = sqlalchemy_url.make_url(self.config.connection_url.get_secret_value())
            if parsed.get_dialect().is_async:
                self._async_sql_resource = AsyncSqlDatabaseResource(self.config)
                await self._async_sql_resource.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Load API — structured mode
    # ------------------------------------------------------------------

    def load(self, **options: Any) -> FrameResult:
        """Load data from the configured backend.

        In **structured mode** (no ``table`` set at construction) pass explicit
        ``statement``, ``model``, ``filters``, etc.

        In **configured mode** (``table`` set at construction) pass runtime
        filter values as keyword arguments.  Control keywords like ``limit``,
        ``as_pandas``, and ``persist`` are still recognised and forwarded.

        Args:
            persist: If ``True`` and the result is a Dask DataFrame, pin it on
                the workers with ``.persist()`` before returning.
            as_pandas: If ``True``, compute the result to a Pandas DataFrame.
            **options: Filter kwargs (configured mode) or load-request fields.
        """
        persist: bool = bool(options.pop("persist", False))
        resilient: bool = bool(options.pop("resilient", False))
        dry_run: bool = bool(options.pop("dry_run", False))
        diagnostics: bool = bool(options.pop("diagnostics", False))
        as_pandas: bool = bool(options.get("as_pandas", False))
        in_chunk_strategy = options.pop("in_chunk_strategy", "auto")
        in_chunk_size_raw = options.pop("in_chunk_size", None)
        in_chunk_concurrency_raw = options.pop("in_chunk_concurrency", None)
        started = perf_counter()
        requested_return_type = self._resolve_requested_return_type(as_pandas=as_pandas, options=options)
        requested_execution_mode = self._resolve_requested_execution_mode(options)
        resolved_return_type, resolved_execution_mode = self._resolve_execution_plan(
            requested_return_type=requested_return_type,
            requested_execution_mode=requested_execution_mode,
            options=options,
        )
        loader_return_type = self._loader_return_type(
            resolved_return_type,
            backend=self.backend,
            execution_mode=resolved_execution_mode,
        )
        loader_as_pandas = self._loader_as_pandas(
            backend=self.backend,
            execution_mode=resolved_execution_mode,
            loader_return_type=loader_return_type,
        )
        if dry_run and (
            resolved_execution_mode != "lazy" or resolved_return_type != "dask"
        ):
            raise ValueError("dry_run=True is only supported for lazy Dask gateway loads.")
        if diagnostics:
            self._log_load_start(
                requested_return_type=requested_return_type,
                resolved_return_type=resolved_return_type,
                requested_execution_mode=requested_execution_mode,
                resolved_execution_mode=resolved_execution_mode,
                loader_return_type=loader_return_type,
                persist=persist,
                resilient=resilient,
                dry_run=dry_run,
            )
        strategy = self._frame_strategy(resolved_return_type)
        loader_options = self._prepare_structured_loader_options(
            {**options, "as_pandas": loader_as_pandas}
        )
        if diagnostics:
            loader_options["diagnostics"] = True

        loader_options = self._resolve_series_filters(loader_options)
        in_chunk_size, in_chunk_concurrency = self._resolve_in_chunk_controls(
            loader_options,
            strategy=in_chunk_strategy,
            in_chunk_size_raw=in_chunk_size_raw,
            in_chunk_concurrency_raw=in_chunk_concurrency_raw,
        )

        def _execute_sync(**opts: Any) -> pd.DataFrame | dd.DataFrame | pa.Table:
            if self._configured:
                return self._load_configured(
                    opts,
                    return_type=resolved_return_type,
                    execution_mode=resolved_execution_mode,
                    loader_return_type=loader_return_type,
                    loader_as_pandas=loader_as_pandas,
                )
            if self.backend == "sqlalchemy":
                assert isinstance(self.config, SqlDatabaseConfig)
                if self.resource is None:
                    raise RuntimeError(
                        "DataGateway.load() requires a synchronous DSN. "
                        "Switch to a sync DSN (e.g. 'mysql+pymysql://...') or use aload() instead."
                    )
                assert isinstance(self.resource, SqlDatabaseResource)
                if resolved_execution_mode == "lazy":
                    return load_sql_partitioned(
                        self.config,
                        self.resource,
                        build_sql_partitioned_request(opts),
                    )
                df_local = load_sql(
                    self.resource,
                    SqlLoadRequest.model_validate(
                        self._structured_sql_request_payload(
                            opts,
                            return_type=loader_return_type,
                        )
                    ),
                )
                if isinstance(df_local, pd.DataFrame) and opts.get("statement") is not None:
                    return self._coerce_eager_sql_frame(
                        df_local,
                        statement=opts["statement"],
                    )
                return df_local
            if self.backend == "parquet":
                return load_parquet(
                    self.resource,
                    ParquetLoadRequest.model_validate(
                        self._structured_parquet_request_payload(
                            opts,
                            return_type=loader_return_type,
                        )
                    ),
                )
            raise RuntimeError(f"Unsupported backend: {self.backend}")

        df = self._chunked_in_load_sync(
            _execute_sync,
            in_chunk_size,
            loader_options,
            return_type=resolved_return_type,
            max_concurrency=in_chunk_concurrency,
        )

        result = strategy.normalize(df)
        if dry_run:
            if diagnostics:
                self._log_load_dry_run(result, elapsed=perf_counter() - started, persist=persist)
            return result
        if persist and isinstance(result, dd.DataFrame):
            result = safe_persist(result, logger=self._logger) if resilient else get_frame_strategy("dask").persist(result)
        if diagnostics:
            self._log_load_complete(result, elapsed=perf_counter() - started)
        return result

    async def aload(self, **options: Any) -> FrameResult:
        """Async variant of :meth:`load`.

        Additional kwargs beyond :meth:`load`:

        Args:
            timeout: Seconds to wait for the backend load before raising
                ``asyncio.TimeoutError``.  ``None`` means no timeout.
            chunk_size: Override for splitting massive ``field__in`` filter
                lists.  Defaults to ``_DEFAULT_IN_CHUNK_SIZE`` (900).
            in_chunk_concurrency: Optional cap on how many chunked ``__in``
                sub-queries run at once. ``None`` preserves the existing
                unbounded fan-out behavior.
        """
        persist: bool = bool(options.pop("persist", False))
        resilient: bool = bool(options.pop("resilient", False))
        dry_run: bool = bool(options.pop("dry_run", False))
        diagnostics: bool = bool(options.pop("diagnostics", False))
        as_pandas: bool = bool(options.get("as_pandas", False))
        timeout: float | None = options.pop("timeout", None)
        in_chunk_strategy = options.pop("in_chunk_strategy", "auto")
        in_chunk_size_raw = options.pop("in_chunk_size", None)
        in_chunk_concurrency_raw = options.pop("in_chunk_concurrency", None)
        started = perf_counter()
        requested_return_type = self._resolve_requested_return_type(as_pandas=as_pandas, options=options)
        requested_execution_mode = self._resolve_requested_execution_mode(options)
        resolved_return_type, resolved_execution_mode = await self._resolve_execution_plan_async(
            requested_return_type=requested_return_type,
            requested_execution_mode=requested_execution_mode,
            options=options,
        )
        loader_return_type = self._loader_return_type(
            resolved_return_type,
            backend=self.backend,
            execution_mode=resolved_execution_mode,
        )
        loader_as_pandas = self._loader_as_pandas(
            backend=self.backend,
            execution_mode=resolved_execution_mode,
            loader_return_type=loader_return_type,
        )
        if dry_run and (
            resolved_execution_mode != "lazy" or resolved_return_type != "dask"
        ):
            raise ValueError("dry_run=True is only supported for lazy Dask gateway loads.")
        if diagnostics:
            self._log_load_start(
                requested_return_type=requested_return_type,
                resolved_return_type=resolved_return_type,
                requested_execution_mode=requested_execution_mode,
                resolved_execution_mode=resolved_execution_mode,
                loader_return_type=loader_return_type,
                persist=persist,
                resilient=resilient,
                dry_run=dry_run,
            )
        strategy = self._frame_strategy(resolved_return_type)

        # Resolve any Series values before chunked dispatch so the chunker
        # sees plain lists (which it already knows how to split).
        loader_options = await self._resolve_series_filters_async(
            self._prepare_structured_loader_options(
                {**options, "as_pandas": loader_as_pandas}
            )
        )
        if diagnostics:
            loader_options["diagnostics"] = True
        in_chunk_size, in_chunk_concurrency = self._resolve_in_chunk_controls(
            loader_options,
            strategy=in_chunk_strategy,
            in_chunk_size_raw=in_chunk_size_raw,
            in_chunk_concurrency_raw=in_chunk_concurrency_raw,
        )

        async def _execute(**opts: Any) -> pd.DataFrame | dd.DataFrame | pa.Table:
            if self._configured:
                coro = self._aload_configured(
                    opts,
                    return_type=resolved_return_type,
                    execution_mode=resolved_execution_mode,
                    loader_return_type=loader_return_type,
                    loader_as_pandas=loader_as_pandas,
                )
            elif self.backend == "sqlalchemy":
                assert isinstance(self.config, SqlDatabaseConfig)
                if resolved_execution_mode == "lazy":
                    request = build_sql_partitioned_request(opts)
                    if self.resource is None:
                        # Async DSN: plan and execute entirely on the async engine.
                        # Opens its own connections; no asyncio.to_thread blocking.
                        async_resource = self._async_sql_resource
                        if async_resource is None:
                            async def _run_with_temp(
                                _req: SqlPartitionedLoadRequest = request,
                            ) -> pd.DataFrame | dd.DataFrame:
                                async with AsyncSqlDatabaseResource(self.config) as _tmp:
                                    return await self._run_async_partitioned_request(_tmp, _req)
                            coro = _run_with_temp()
                        else:
                            coro = self._run_async_partitioned_request(async_resource, request)
                    else:
                        assert isinstance(self.resource, SqlDatabaseResource)
                        coro = asyncio.to_thread(
                            load_sql_partitioned, self.config, self.resource, request
                        )
                else:
                    request = SqlLoadRequest.model_validate(
                        self._structured_sql_request_payload(
                            opts,
                            return_type=loader_return_type,
                        )
                    )
                    coro = self._aload_sql(request)
                    if opts.get("statement") is not None:
                        async def _coerce_sql_coro(
                            _coro=coro,
                            _statement=opts["statement"],
                        ) -> pd.DataFrame | pa.Table:
                            frame = await _coro
                            if not isinstance(frame, pd.DataFrame):
                                return frame
                            return self._coerce_eager_sql_frame(frame, statement=_statement)
                        coro = _coerce_sql_coro()
            elif self.backend == "parquet":
                request = ParquetLoadRequest.model_validate(
                    self._structured_parquet_request_payload(
                        opts,
                        return_type=loader_return_type,
                    )
                )
                coro = asyncio.to_thread(load_parquet, self.resource, request)
            else:
                raise RuntimeError(f"Unsupported backend: {self.backend}")
            if timeout is not None:
                return await asyncio.wait_for(coro, timeout)
            return await coro

        df = await self._chunked_in_load(
            _execute,
            in_chunk_size,
            loader_options,
            return_type=resolved_return_type,
            max_concurrency=in_chunk_concurrency,
        )
        result = strategy.normalize(df)
        if dry_run:
            if diagnostics:
                self._log_load_dry_run(result, elapsed=perf_counter() - started, persist=persist)
            return result
        if persist and isinstance(result, dd.DataFrame):
            result = safe_persist(result, logger=self._logger) if resilient else get_frame_strategy("dask").persist(result)
        if diagnostics:
            self._log_load_complete(result, elapsed=perf_counter() - started)
        return result

    def preview(
        self,
        *,
        n: int = 5,
        npartitions: int = 1,
        **options: Any,
    ) -> FrameResult:
        """Load a frame and return a small preview with Dask-safe head semantics."""

        frame = self.load(**options)
        return self._preview_frame(frame, n=n, npartitions=npartitions)

    async def apreview(
        self,
        *,
        n: int = 5,
        npartitions: int = 1,
        **options: Any,
    ) -> FrameResult:
        """Async companion to :meth:`preview`."""

        frame = await self.aload(**options)
        return await self._apreview_frame(frame, n=n, npartitions=npartitions)

    def load_period(
        self,
        dt_field: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> FrameResult:
        """Load rows within a date range.

        Args:
            dt_field: Semantic column name of the datetime/date field.
            start: ISO-8601 start date (inclusive).
            end: ISO-8601 end date (inclusive).
            **kwargs: Forwarded to :meth:`load` as control or filter kwargs.
        """
        return self.load(**self._prepare_period_filters(dt_field, start, end, **kwargs))

    async def aload_period(
        self,
        dt_field: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> FrameResult:
        """Async variant of :meth:`load_period`."""
        return await self.aload(**self._prepare_period_filters(dt_field, start, end, **kwargs))

    def _prepare_period_filters(
        self, dt_field: str, start: str, end: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Build filter kwargs for a date-range query.

        ``dt_field`` is always expressed as a **semantic name**.  When a
        ``field_map`` is configured the gateway translates it to the DB column
        name automatically during the load — no manual mapping needed here.

        Uses direct ``__exact`` / ``__gte`` + ``__lte`` comparison so the
        filters work correctly for DATE, TIMESTAMP, and ISO-text columns across
        all supported databases.  If you specifically need date-part extraction
        from a full DATETIME column, pass ``field__date__gte`` / ``__lte``
        filters directly instead.
        """
        return prepare_period_filters(dt_field, start, end, **kwargs)

    async def _chunked_in_load(
        self,
        execute_fn: Any,
        chunk_size: int,
        options: dict[str, Any],
        *,
        return_type: ReturnType,
        max_concurrency: int | None = None,
    ) -> FrameResult:
        """Split a massive ``field__in=[...]`` filter into concurrent batches.

        If no ``__in`` key exceeds *chunk_size*, delegates straight to
        *execute_fn*.  Otherwise fires one task per chunk, optionally bounded
        by *max_concurrency*, and concatenates the results.
        """
        if chunk_size <= 0:
            return await execute_fn(**options)

        chunk_target = self._find_chunk_target(options, chunk_size=chunk_size)
        if chunk_target is None:
            return await execute_fn(**options)

        tasks = []
        semaphore: asyncio.Semaphore | None = None
        if max_concurrency is not None and max_concurrency > 0:
            semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_chunk(chunk_opts: dict[str, Any]) -> FrameResult:
            if semaphore is None:
                return await execute_fn(**chunk_opts)
            async with semaphore:
                return await execute_fn(**chunk_opts)

        for chunk_opts in self._iter_chunk_option_sets(options, chunk_target, chunk_size):
            tasks.append(_run_chunk(chunk_opts))

        strategy = self._frame_strategy(return_type)
        results: list[FrameResult] = await asyncio.gather(*tasks)
        non_empty = [r for r in results if self._has_any_rows(r)]
        if not non_empty:
            return results[0]
        if len(non_empty) == 1:
            return non_empty[0]
        combined = strategy.concat(non_empty)
        if isinstance(combined, dd.DataFrame):
            return get_frame_strategy("dask").persist(combined)
        return combined

    def _chunked_in_load_sync(
        self,
        execute_fn: Any,
        chunk_size: int,
        options: dict[str, Any],
        *,
        return_type: ReturnType,
        max_concurrency: int | None = None,
    ) -> FrameResult:
        """Sync variant of :meth:`_chunked_in_load`.

        Chunking is only activated when the caller explicitly opts in on the
        sync path, because this may issue multiple queries instead of one.
        """
        if chunk_size <= 0:
            return execute_fn(**options)

        chunk_target = self._find_chunk_target(options, chunk_size=chunk_size)
        if chunk_target is None:
            return execute_fn(**options)

        chunk_opts_list = list(self._iter_chunk_option_sets(options, chunk_target, chunk_size))

        if max_concurrency is not None and max_concurrency > 1:
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                futures = [executor.submit(execute_fn, **chunk_opts) for chunk_opts in chunk_opts_list]
                results = [future.result() for future in futures]
        else:
            results = [execute_fn(**chunk_opts) for chunk_opts in chunk_opts_list]

        strategy = self._frame_strategy(return_type)
        non_empty = [r for r in results if self._has_any_rows(r)]
        if not non_empty:
            return results[0]
        if len(non_empty) == 1:
            return non_empty[0]
        combined = strategy.concat(non_empty)
        if isinstance(combined, dd.DataFrame):
            return get_frame_strategy("dask").persist(combined)
        return combined

    @staticmethod
    def _is_chunkable_in_value(value: Any, *, chunk_size: int) -> bool:
        return (
            hasattr(value, "__len__")
            and hasattr(value, "__getitem__")
            and not isinstance(value, (str, bytes, pd.Series, dd.Series, pl.Series))
            and len(value) > chunk_size
        )

    def _find_chunk_target(
        self,
        options: dict[str, Any],
        *,
        chunk_size: int,
    ) -> tuple[str, str, Any] | None:
        for key, value in options.items():
            if key in _LOAD_CONTROL_KEYS:
                continue
            if key.endswith("__in") and self._is_chunkable_in_value(value, chunk_size=chunk_size):
                return ("top_level", key, value)

        nested_filters = options.get("filters")
        if isinstance(nested_filters, dict):
            for key, value in nested_filters.items():
                if key.endswith("__in") and self._is_chunkable_in_value(value, chunk_size=chunk_size):
                    return ("filters", key, value)
        return None

    def _iter_chunk_option_sets(
        self,
        options: dict[str, Any],
        chunk_target: tuple[str, str, Any],
        chunk_size: int,
    ) -> Any:
        scope, key, values = chunk_target
        for i in range(0, len(values), chunk_size):
            chunk_opts = dict(options)
            if scope == "top_level":
                chunk_opts[key] = values[i : i + chunk_size]
            else:
                nested_filters = dict(chunk_opts.get("filters", {}))
                nested_filters[key] = values[i : i + chunk_size]
                chunk_opts["filters"] = nested_filters
            yield chunk_opts

    def _resolve_in_chunk_controls(
        self,
        options: dict[str, Any],
        *,
        strategy: str,
        in_chunk_size_raw: Any,
        in_chunk_concurrency_raw: Any,
    ) -> tuple[int, int | None]:
        if in_chunk_size_raw is not None or in_chunk_concurrency_raw is not None:
            size = _DEFAULT_IN_CHUNK_SIZE if in_chunk_size_raw is None else int(in_chunk_size_raw)
            concurrency = None if in_chunk_concurrency_raw is None else int(in_chunk_concurrency_raw)
            return size, concurrency

        if strategy == "off":
            return 0, None
        if strategy != "auto":
            raise ValueError("in_chunk_strategy must be 'auto' or 'off'.")

        filters = self._extract_filter_mapping(options)
        if self.backend == "sqlalchemy" and filters:
            hint = self._filter_chunking_hint(filters)
            if hint is not None:
                return int(hint["in_chunk_size"]), int(hint["in_chunk_concurrency"])

        return _DEFAULT_IN_CHUNK_SIZE, None

    def _extract_filter_mapping(self, options: dict[str, Any]) -> dict[str, Any]:
        nested_filters = options.get("filters")
        if isinstance(nested_filters, dict):
            return nested_filters
        _, runtime_filters = split_control_and_filters(options)
        return runtime_filters

    @staticmethod
    def _filter_chunking_hint(filters: dict[str, Any]) -> dict[str, Any] | None:
        return FilterHandler("sqlalchemy").suggest_sql_in_chunking(filters)

    def _supports_lazy_series_semi_join(
        self,
        join_series: pd.Series | dd.Series | pl.Series,
        on: str,
        options: dict[str, Any],
    ) -> bool:
        if options.get("as_pandas"):
            return False
        if options.get("execution_mode") == "eager" or options.get("partitioned") is False:
            return False
        requested_return_type = options.get(
            "return_type",
            self._return_type if self._configured else "dask",
        )
        if requested_return_type != "dask":
            return False
        if self._configured:
            return not self._df_params.fieldnames or on in self._df_params.fieldnames
        statement = options.get("statement")
        if statement is None:
            return False
        selected_names = [
            str(getattr(selected, "key", None) or getattr(selected, "name", None))
            for selected in statement.selected_columns
        ]
        return on in selected_names

    @staticmethod
    def _series_to_dask_key_frame(
        join_series: pd.Series | dd.Series | pl.Series,
        *,
        column_name: str,
    ) -> dd.DataFrame:
        if isinstance(join_series, dd.Series):
            return join_series.dropna().drop_duplicates().to_frame(name=column_name)
        if isinstance(join_series, pl.Series):
            join_series = pd.Series(join_series.to_list(), name=column_name)
        if isinstance(join_series, pd.Series):
            return dd.from_pandas(
                join_series.dropna().drop_duplicates().to_frame(name=column_name),
                npartitions=1,
            )
        raise TypeError(f"Unsupported join series type: {type(join_series)!r}")

    def _lazy_series_semi_join(
        self,
        join_series: pd.Series | dd.Series | pl.Series,
        *,
        on: str,
        options: dict[str, Any],
    ) -> dd.DataFrame:
        base_options = {
            key: value
            for key, value in options.items()
            if key not in {"return_type", "execution_mode", "as_pandas", f"{on}__in"}
        }
        frame = self.load(
            **base_options,
            return_type="dask",
            execution_mode="lazy",
        )
        assert isinstance(frame, dd.DataFrame)
        key_frame = self._series_to_dask_key_frame(join_series, column_name=on)
        if on in frame.columns:
            key_frame = key_frame.astype({on: frame.dtypes[on]})
        joined = frame.merge(key_frame, how="inner", on=on)
        return joined[list(frame.columns)]

    @staticmethod
    def _has_any_rows(df: FrameResult) -> bool:
        """Return ``True`` if *df* contains at least one row."""
        try:
            return DataGateway._strategy_for_frame(df).has_any_rows(df)
        except Exception:
            return False

    def semi_join(
        self,
        join_series: pd.Series | dd.Series | pl.Series,
        on: str,
        **kwargs: Any,
    ) -> FrameResult:
        """Load rows whose *on* column value appears in *join_series*.

        This is the primary entry point for the **distributed semi-join** pattern:
        callers can pass a Dask or pandas Series of key values and the gateway
        resolves it to a deduplicated list before issuing the query.

        Syntactic sugar for ``load(field__in=join_series, ...)``.

        Args:
            join_series: Pandas, Dask, or Polars Series of key values to filter by.
                Duplicate values and ``NaN`` are silently discarded.
            on: Semantic column name to match against (translated through the
                ``field_map`` automatically when one is configured).
            **kwargs: Additional filter / control kwargs forwarded to
                :meth:`load`.
        """
        if self._supports_lazy_series_semi_join(join_series, on, kwargs):
            return self._lazy_series_semi_join(join_series, on=on, options=kwargs)
        kwargs[f"{on}__in"] = join_series
        return self.load(**kwargs)

    async def asemi_join(
        self,
        join_series: pd.Series | dd.Series | pl.Series,
        on: str,
        **kwargs: Any,
    ) -> FrameResult:
        """Async variant of :meth:`semi_join`."""
        if self._supports_lazy_series_semi_join(join_series, on, kwargs):
            return await asyncio.to_thread(
                self._lazy_series_semi_join,
                join_series,
                on=on,
                options=kwargs,
            )
        kwargs[f"{on}__in"] = join_series
        return await self.aload(**kwargs)

    @staticmethod
    def _resolve_series_filters(options: dict[str, Any]) -> dict[str, Any]:
        """Replace ``field__in=Series`` values with deduplicated plain lists.

        Pandas, Dask, and Polars Series are
        accepted.  Dask Series are synchronously computed — use
        :meth:`_resolve_series_filters_async` in async contexts.

        This is the pre-processing step for the distributed semi-join pattern:
        callers pass a Series of IDs as a ``field__in`` filter and the gateway
        automatically resolves it before issuing the query.  The resolved list
        is then handled by the existing chunked-IN machinery.
        """
        strategy = get_frame_strategy("dask")
        if not any(
            k.endswith("__in") and isinstance(v, (pd.Series, dd.Series, pl.Series))
            for k, v in options.items()
        ):
            return options
        resolved = dict(options)
        for key, value in options.items():
            if key.endswith("__in") and isinstance(value, (pd.Series, dd.Series, pl.Series)):
                resolved[key] = strategy.resolve_series(value)
        return resolved

    @staticmethod
    async def _resolve_series_filters_async(options: dict[str, Any]) -> dict[str, Any]:
        """Async variant of :meth:`_resolve_series_filters`.

        Dask Series are computed in a thread pool so the event loop is not
        blocked while Dask executes the computation graph.
        """
        dask_keys = [
            k
            for k, v in options.items()
            if k.endswith("__in") and isinstance(v, dd.Series)
        ]
        pandas_or_polars_keys = [
            k
            for k, v in options.items()
            if k.endswith("__in") and isinstance(v, (pd.Series, pl.Series)) and not isinstance(v, dd.Series)
        ]
        if not dask_keys and not pandas_or_polars_keys:
            return options
        resolved = dict(options)

        if dask_keys:
            computed = await asyncio.to_thread(
                dask.compute,
                *[options[key] for key in dask_keys],
            )
            for key, series in zip(dask_keys, computed):
                resolved[key] = series.dropna().unique().tolist()

        strategy = get_frame_strategy("dask")
        for key in pandas_or_polars_keys:
            resolved[key] = strategy.resolve_series(options[key])
        return resolved

    async def aclose(self) -> None:
        if self._async_sql_resource is not None:
            await self._async_sql_resource.__aexit__(None, None, None)
            self._async_sql_resource = None
        self.close()

    async def _aload_sql(self, request: SqlLoadRequest) -> pd.DataFrame | dd.DataFrame:
        async_resource = self._async_sql_resource
        if async_resource is None:
            try:
                assert isinstance(self.config, SqlDatabaseConfig)
                async with AsyncSqlDatabaseResource(self.config) as temporary_resource:
                    return await read_sql_async(temporary_resource, request)
            except SQLAlchemyError:
                if self.resource is None:
                    raise
                assert isinstance(self.resource, SqlDatabaseResource)
                return await asyncio.to_thread(load_sql, self.resource, request)
        return await read_sql_async(async_resource, request)

    async def _run_async_partitioned_request(
        self,
        async_resource: Any,
        request: SqlPartitionedLoadRequest,
    ) -> pd.DataFrame | dd.DataFrame | pa.Table:
        """Execute a pre-built ``SqlPartitionedLoadRequest`` via the async engine.

        Planning queries (COUNT, MIN/MAX) run directly on the async engine,
        avoiding ``MissingGreenlet`` errors.  For range partitioning they are
        issued in parallel over two independent connections.
        """
        assert isinstance(self.config, SqlDatabaseConfig)

        # SqlPartitionPlanner needs .engine only for dialect (compile_partition).
        # Accessing .sync_engine is safe here: it is a property read, not a
        # connection open, so no greenlet context is required.
        class _EngineAdapter:
            engine = async_resource.engine.sync_engine
            logger = async_resource.logger
            debug = False

        started = perf_counter()
        planner = SqlPartitionPlanner(_EngineAdapter())  # type: ignore[arg-type]
        plan = await planner.async_plan_request(request, async_resource.engine)
        if request.diagnostics:
            async_resource.logger.info(
                "Partitioned SQL plan "
                f"strategy={plan.strategy} partitions={len(plan.partitions)} rows={plan.total_rows} "
                f"chunk_size={request.chunk_size} max_concurrent_fetches={request.max_concurrent_fetches} "
                f"use_arrow={request.use_arrow}"
            )

        worker_config = WorkerSqlConfig.from_database_config(self.config)
        gate_key = _get_worker_engine_identity(worker_config)
        executor = SqlPartitionExecutor(worker_config, gate_key)
        result = executor.load_plan(
            plan,
            as_pandas=request.as_pandas,
            max_concurrent_fetches=request.max_concurrent_fetches,
            fetch_partition=SqlPartitionExecutor.fetch_partition_async,
        )
        if request.diagnostics:
            result_partitions = result.npartitions if isinstance(result, dd.DataFrame) else 1
            async_resource.logger.info(
                "Partitioned SQL load completed "
                f"strategy={plan.strategy} partitions={len(plan.partitions)} "
                f"rows={plan.total_rows} result_partitions={result_partitions} "
                f"elapsed={perf_counter() - started:.2f}s"
            )
        return result

    async def _run_async_partitioned(
        self,
        async_resource: Any,
        model: Any,
        stmt: Any,
        db_filters: dict[str, Any],
        control: dict[str, Any],
    ) -> pd.DataFrame | dd.DataFrame:
        """Plan and execute a configured-mode partitioned load via the async engine."""
        partitioned_options = build_partitioned_load_options(
            statement=stmt,
            model=model,
            filters=db_filters,
            control=control,
            default_chunk_size=self._df_params.chunk_size,
        )
        request = build_sql_partitioned_request(partitioned_options)
        return await self._run_async_partitioned_request(async_resource, request)

    # ------------------------------------------------------------------
    # Configured-mode internals
    # ------------------------------------------------------------------

    def _build_configured_request(
        self, options: dict[str, Any]
    ) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
        """Return ``(model, statement, db_filters, control_kwargs)`` for configured mode.

        All filter inputs (``sticky_filters``, runtime kwargs, explicit ``filters=``)
        are always expressed as **semantic names**.

        When a ``field_map`` is configured the underlying DB uses non-semantic
        column names, so semantic names are translated to DB column names before
        the query is built and results are renamed back to semantic names.  When
        no ``field_map`` is given the DB already uses semantic names and no
        translation is performed.

        Parquet backends never need translation (files already carry semantic
        column names) — this path is never reached for parquet.
        """
        assert isinstance(self.resource, SqlDatabaseResource)

        normalized = normalize_configured_filters(
            options,
            sticky_filters=self._sticky_filters,
            exclude=self._exclude,
        )
        control = normalized.control
        combined_filters = normalized.filters
        configured_fieldnames = self._configured_fieldnames(control)

        # When a field_map is present the DB has non-semantic column names;
        # translate semantic inputs → DB column names.
        # When absent the DB already uses semantic names; pass through as-is.
        if self._field_map:
            db_filters = self._field_map.translate_filters_to_db(
                combined_filters, input_keys_are="semantic"
            )
            db_columns: list[str] | None = (
                self._field_map.select_db_columns(configured_fieldnames)
                if configured_fieldnames
                else None
            )
        else:
            db_filters = combined_filters
            db_columns = (
                list(configured_fieldnames)
                if configured_fieldnames
                else None
            )

        model, stmt = self._get_configured_select(db_columns)

        return model, stmt, db_filters, control

    def _log_load_start(
        self,
        *,
        requested_return_type: ReturnType,
        resolved_return_type: ResolvedReturnType,
        requested_execution_mode: ExecutionMode,
        resolved_execution_mode: ResolvedExecutionMode,
        loader_return_type: Literal["pandas", "arrow", "dask"],
        persist: bool,
        resilient: bool,
        dry_run: bool,
    ) -> None:
        logger = self._logger
        if logger is None:
            return
        if hasattr(logger, "set_level"):
            logger.set_level(Logger.INFO)
        logger.info(
            "Gateway load starting "
            f"backend={self.backend} configured={self._configured} "
            f"requested_return_type={requested_return_type} "
            f"resolved_return_type={resolved_return_type} "
            f"requested_execution_mode={requested_execution_mode} "
            f"resolved_execution_mode={resolved_execution_mode} "
            f"loader_return_type={loader_return_type} "
            f"persist={persist} "
            f"resilient={resilient} "
            f"dry_run={dry_run}"
        )
        client_summary = current_client_summary()
        if client_summary is not None:
            logger.info(f"Gateway load active Dask client={client_summary}")

    def _log_load_complete(self, frame: FrameResult, *, elapsed: float) -> None:
        logger = self._logger
        if logger is None:
            return
        if hasattr(logger, "set_level"):
            logger.set_level(Logger.INFO)
        if isinstance(frame, dd.DataFrame):
            logger.info(f"Gateway load graph metrics={inspect_graph(frame)}")
        logger.info(
            f"Gateway load completed elapsed={elapsed:.2f}s metrics={describe_frame(frame)}"
        )

    def _log_load_dry_run(
        self,
        frame: FrameResult,
        *,
        elapsed: float,
        persist: bool,
    ) -> None:
        logger = self._logger
        if logger is None:
            return
        if hasattr(logger, "set_level"):
            logger.set_level(Logger.INFO)
        if isinstance(frame, dd.DataFrame):
            logger.info(f"Gateway load dry run graph metrics={inspect_graph(frame)}")
        if persist:
            logger.info("Gateway load dry run skipped persist=True materialization.")
        logger.info(
            f"Gateway load dry run completed elapsed={elapsed:.2f}s metrics={describe_frame(frame)}"
        )

    def _preview_frame(
        self,
        frame: FrameResult,
        *,
        n: int,
        npartitions: int,
    ) -> FrameResult:
        if isinstance(frame, dd.DataFrame):
            return safe_head(frame, n=n, npartitions=npartitions, logger=self._logger)
        if isinstance(frame, pd.DataFrame):
            return frame.head(n)
        if isinstance(frame, pa.Table):
            return frame.slice(0, n)
        if isinstance(frame, pl.DataFrame):
            return frame.head(n)
        raise TypeError(f"Unsupported frame type: {type(frame)!r}")

    async def _apreview_frame(
        self,
        frame: FrameResult,
        *,
        n: int,
        npartitions: int,
    ) -> FrameResult:
        if isinstance(frame, dd.DataFrame):
            return await async_safe_head(
                frame,
                n=n,
                npartitions=npartitions,
                logger=self._logger,
            )
        return self._preview_frame(frame, n=n, npartitions=npartitions)

    @staticmethod
    def _coerce_eager_sql_frame(frame: pd.DataFrame, *, statement: Any) -> pd.DataFrame:
        meta_dtypes = SqlPartitionPlanner.infer_meta_dtypes(statement)
        return apply_schema_map(frame, meta_dtypes, require_columns=True)

    def _configured_fieldnames(self, control: dict[str, Any]) -> tuple[str, ...] | None:
        runtime_columns = control.get("columns")
        if runtime_columns is not None:
            return tuple(runtime_columns)
        return self._df_params.fieldnames

    @staticmethod
    def _trusted_sql_request(
        *,
        sql: str | None = None,
        statement: Any = None,
        model: Any = None,
        filters: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        limit: int | None = None,
        columns: list[str] | None = None,
        as_pandas: bool = False,
        diagnostics: bool = False,
        return_type: ResolvedReturnType = "pandas",
    ) -> SqlLoadRequest:
        return SqlLoadRequest.model_construct(
            sql=sql,
            statement=statement,
            model=model,
            filters=filters or {},
            params=params or {},
            limit=limit,
            columns=columns,
            as_pandas=as_pandas,
            diagnostics=diagnostics,
            return_type=return_type,
        )

    @staticmethod
    def _trusted_parquet_request(
        *,
        filters: dict[str, Any] | None = None,
        raw_filters: list[Any] | None = None,
        limit: int | None = None,
        columns: list[str] | None = None,
        as_pandas: bool = False,
        diagnostics: bool = False,
        return_type: ResolvedReturnType = "pandas",
    ) -> ParquetLoadRequest:
        return ParquetLoadRequest.model_construct(
            filters=filters or {},
            raw_filters=raw_filters,
            limit=limit,
            columns=columns,
            as_pandas=as_pandas,
            diagnostics=diagnostics,
            return_type=return_type,
        )

    def _apply_field_map(
        self,
        frame: FrameResult,
        *,
        strategy: FrameStrategy | None = None,
    ) -> FrameResult:
        """Rename DB column names to semantic names.

        The rename only fires when a ``field_map`` is configured.
        ``column_names`` positional renaming is intentionally kept separate so
        that :meth:`_filter_to_fieldnames` can run against semantic names before
        the positional override obliterates them.
        """
        strategy = strategy or self._strategy_for_frame(frame)
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
        """Apply the positional ``column_names`` rename if configured."""
        strategy = strategy or self._strategy_for_frame(frame)
        if self._df_params.column_names:
            frame = strategy.apply_column_names(frame, self._df_params.column_names)
        return frame

    def _filter_to_fieldnames(
        self,
        frame: FrameResult,
        *,
        strategy: FrameStrategy | None = None,
        fieldnames: tuple[str, ...] | None = None,
    ) -> FrameResult:
        """Narrow the DataFrame to the requested ``fieldnames`` columns.

        For the SQL backend this is normally a no-op because ``fieldnames``
        already drives the SELECT clause.  For the parquet backend (and as a
        safety net elsewhere) it performs the post-load column selection.

        A warning is logged for any requested column that is absent from the
        result so callers notice schema drift early.
        """
        strategy = strategy or self._strategy_for_frame(frame)
        fieldnames = fieldnames if fieldnames is not None else self._df_params.fieldnames
        if not fieldnames:
            return frame

        present = set(frame.column_names if isinstance(frame, pa.Table) else frame.columns)
        missing = [f for f in fieldnames if f not in present]
        if missing:
            import warnings
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
        strategy = strategy or self._strategy_for_frame(frame)
        return strategy.apply_options(frame, self._df_params, self._df_options)

    def _load_configured(
        self,
        options: dict[str, Any],
        *,
        return_type: ReturnType,
        execution_mode: ResolvedExecutionMode,
        loader_return_type: Literal["pandas", "arrow", "dask"],
        loader_as_pandas: bool,
    ) -> FrameResult:
        # Parquet: files already carry semantic column names — pass filters
        # through unchanged and skip field_map rename entirely.
        if self.backend == "parquet":
            assert isinstance(self.resource, ParquetDataResource)
            normalized = normalize_configured_filters(
                options,
                sticky_filters=self._sticky_filters,
                exclude=self._exclude,
            )
            configured_fieldnames = self._configured_fieldnames(normalized.control)
            df = load_parquet(
                self.resource,
                self._trusted_parquet_request(
                    filters=normalized.filters,
                    raw_filters=normalized.control.get("raw_filters"),
                    limit=normalized.control.get("limit"),
                    columns=list(configured_fieldnames) if configured_fieldnames else None,
                    as_pandas=loader_as_pandas,
                    diagnostics=bool(normalized.control.get("diagnostics", False)),
                    return_type=loader_return_type,
                ),
            )
            return self._finalize_configured_result(
                df,
                return_type=return_type,
                apply_field_map=False,
                fieldnames=configured_fieldnames,
            )

        assert isinstance(self.config, SqlDatabaseConfig)
        if self.resource is None:
            raise RuntimeError(
                "Configured-mode SQL loads require a synchronous DSN. "
                "Switch to a sync DSN (e.g. 'mysql+pymysql://...') or use aload() instead."
            )
        assert isinstance(self.resource, SqlDatabaseResource)

        model, stmt, db_filters, control = self._build_configured_request(options)
        configured_fieldnames = self._configured_fieldnames(control)
        if execution_mode == "lazy":
            partitioned_options = build_partitioned_load_options(
                statement=stmt,
                model=model,
                filters=db_filters,
                control=control,
                default_chunk_size=self._df_params.chunk_size,
            )
            df = load_sql_partitioned(
                self.config,
                self.resource,
                build_sql_partitioned_request(partitioned_options),
            )
        else:
            df = load_sql(
                self.resource,
                self._trusted_sql_request(
                    statement=stmt,
                    model=model,
                    filters=db_filters,
                    params=control.get("params", {}),
                    limit=control.get("limit"),
                    as_pandas=True,
                    diagnostics=bool(control.get("diagnostics", False)),
                    return_type=loader_return_type,
                ),
            )
            if isinstance(df, pd.DataFrame):
                df = self._coerce_eager_sql_frame(df, statement=stmt)
        # order: DB→semantic rename → filter fieldnames → positional rename → options
        return self._finalize_configured_result(
            df,
            return_type=return_type,
            apply_field_map=True,
            fieldnames=configured_fieldnames,
        )

    async def _aload_configured(
        self,
        options: dict[str, Any],
        *,
        return_type: ReturnType,
        execution_mode: ResolvedExecutionMode,
        loader_return_type: Literal["pandas", "arrow", "dask"],
        loader_as_pandas: bool,
    ) -> FrameResult:
        # Parquet: no translation needed; skip field_map rename.
        if self.backend == "parquet":
            assert isinstance(self.resource, ParquetDataResource)
            normalized = normalize_configured_filters(
                options,
                sticky_filters=self._sticky_filters,
                exclude=self._exclude,
            )
            configured_fieldnames = self._configured_fieldnames(normalized.control)
            request = self._trusted_parquet_request(
                filters=normalized.filters,
                raw_filters=normalized.control.get("raw_filters"),
                limit=normalized.control.get("limit"),
                columns=list(configured_fieldnames) if configured_fieldnames else None,
                as_pandas=loader_as_pandas,
                diagnostics=bool(normalized.control.get("diagnostics", False)),
                return_type=loader_return_type,
            )
            df = await asyncio.to_thread(load_parquet, self.resource, request)
            return self._finalize_configured_result(
                df,
                return_type=return_type,
                apply_field_map=False,
                fieldnames=configured_fieldnames,
            )

        assert isinstance(self.config, SqlDatabaseConfig)
        if self.resource is None:
            # Async DSN path: reflect via async engine, plan via sync_engine wrapper,
            # execute partitions in Dask worker threads using asyncio.run().
            normalized = normalize_configured_filters(
                options,
                sticky_filters=self._sticky_filters,
                exclude=self._exclude,
            )
            control = normalized.control
            combined_filters = normalized.filters
            configured_fieldnames = self._configured_fieldnames(control)

            if self._field_map:
                db_filters = self._field_map.translate_filters_to_db(
                    combined_filters, input_keys_are="semantic"
                )
                db_columns: list[str] | None = (
                    self._field_map.select_db_columns(configured_fieldnames)
                    if configured_fieldnames
                    else None
                )
            else:
                db_filters = combined_filters
                db_columns = (
                    list(configured_fieldnames)
                    if configured_fieldnames
                    else None
                )

            async_resource = self._async_sql_resource
            if async_resource is None:
                async with AsyncSqlDatabaseResource(self.config) as temp_resource:
                    model, stmt = await self._get_configured_select_async(
                        temp_resource, db_columns
                    )
                    if execution_mode == "lazy":
                        df = await self._run_async_partitioned(
                            temp_resource, model, stmt, db_filters, control
                        )
                    else:
                        df = await read_sql_async(
                            temp_resource,
                            self._trusted_sql_request(
                                statement=stmt,
                                model=model,
                                filters=db_filters,
                                params=control.get("params", {}),
                                limit=control.get("limit"),
                                as_pandas=True,
                                diagnostics=bool(control.get("diagnostics", False)),
                                return_type=loader_return_type,
                            ),
                        )
                        if isinstance(df, pd.DataFrame):
                            df = self._coerce_eager_sql_frame(df, statement=stmt)
            else:
                model, stmt = await self._get_configured_select_async(
                    async_resource, db_columns
                )
                if execution_mode == "lazy":
                    df = await self._run_async_partitioned(
                        async_resource, model, stmt, db_filters, control
                    )
                else:
                    df = await read_sql_async(
                        async_resource,
                        self._trusted_sql_request(
                            statement=stmt,
                            model=model,
                            filters=db_filters,
                            params=control.get("params", {}),
                            limit=control.get("limit"),
                            as_pandas=True,
                            diagnostics=bool(control.get("diagnostics", False)),
                            return_type=loader_return_type,
                        ),
                        )
                    if isinstance(df, pd.DataFrame):
                        df = self._coerce_eager_sql_frame(df, statement=stmt)

            return self._finalize_configured_result(
                df,
                return_type=return_type,
                apply_field_map=True,
                fieldnames=configured_fieldnames,
            )

        assert isinstance(self.resource, SqlDatabaseResource)

        model, stmt, db_filters, control = self._build_configured_request(options)
        configured_fieldnames = self._configured_fieldnames(control)

        partitioned_options = build_partitioned_load_options(
            statement=stmt,
            model=model,
            filters=db_filters,
            control=control,
            default_chunk_size=self._df_params.chunk_size,
        )

        if execution_mode == "lazy":
            request = build_sql_partitioned_request(partitioned_options)
            df = await asyncio.to_thread(
                load_sql_partitioned,
                self.config,
                self.resource,
                request,
            )
        else:
            df = await asyncio.to_thread(
                load_sql,
                self.resource,
                self._trusted_sql_request(
                    statement=stmt,
                    model=model,
                    filters=db_filters,
                    params=control.get("params", {}),
                    limit=control.get("limit"),
                    as_pandas=True,
                    diagnostics=bool(control.get("diagnostics", False)),
                    return_type=loader_return_type,
                )
            )
            if isinstance(df, pd.DataFrame):
                df = self._coerce_eager_sql_frame(df, statement=stmt)
        return self._finalize_configured_result(
            df,
            return_type=return_type,
            apply_field_map=True,
            fieldnames=configured_fieldnames,
        )

    def _finalize_configured_result(
        self,
        result: FrameResult,
        *,
        return_type: ReturnType,
        apply_field_map: bool,
        fieldnames: tuple[str, ...] | None = None,
    ) -> FrameResult:
        strategy = self._frame_strategy(return_type)
        frame = strategy.normalize(result)
        if apply_field_map:
            frame = self._apply_field_map(frame, strategy=strategy)
        return self._apply_df_options(
            self._apply_column_names(
                self._filter_to_fieldnames(frame, strategy=strategy, fieldnames=fieldnames),
                strategy=strategy,
            ),
            strategy=strategy,
        )
