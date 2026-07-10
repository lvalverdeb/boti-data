"""
Dask-first gateway over existing boti_data backend resources.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any, Literal

import dask.dataframe as dd
import fsspec
import pandas as pd
import polars as pl
import pyarrow as pa
from boti_dask import async_safe_head, safe_head, safe_persist  # noqa: F401

from boti_data.db import SqlDatabaseResource
from boti_data.field_map import FieldMap

from . import _factory, _series_filters
from ._backend_strategies import (
    BackendStrategy,
    StructuredLoadContext,
    for_config,
)
from ._backend_strategies import (
    get as get_strategy,
)
from .chunking import (
    ChunkedLoadExecutor,
    InChunkPlanner,
)
from .configured_load import ConfiguredLoadService
from .frame_strategies import FrameResult
from .load_request import GatewayLoadRequest
from .loaders import (
    reflect_and_select,
    reflect_and_select_async,
)
from .normalization import prepare_period_filters, split_control_and_filters
from .planning import LoadPlanner
from .policies import GatewayPolicies
from .post_process import PostProcessor, strategy_for_frame
from .requests import (
    BackendConfig,
    DataFrameOptions,
    DataFrameParams,
    ExecutionMode,
    ResolvedExecutionMode,
    ResolvedReturnType,
    ReturnType,
)
from .return_type import AutoReturnTypeResolver
from .sql_guard import validate_raw_sql_statement

_log = logging.getLogger(__name__)


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

    _CACHE_MAXSIZE = 128

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
        raw_sql_policy: str | None = None,
        policies: GatewayPolicies | None = None,
        strict_filter_validation: bool = False,
        allowed_filter_fields: set[str] | None = None,
        require_datacube_request_validator: bool = False,
    ) -> None:
        config = _factory.coerce_backend_config(config)
        self.config = config
        self._strategy: BackendStrategy = for_config(config)
        self.backend, self.resource = self._strategy.build_resource(
            config,
            fs=fs,
            fs_factory=fs_factory,
        )

        self._async_sql_resource = None

        # Configured-mode state
        self._table = table
        self._field_map = FieldMap(field_map) if field_map else FieldMap({})
        self._sticky_filters: dict[str, Any] = dict(sticky_filters or {})
        self._exclude = exclude
        self._df_params = df_params or DataFrameParams()
        self._df_options = df_options or DataFrameOptions()
        self._return_type: ReturnType = self._df_params.return_type
        self._execution_mode: ExecutionMode = self._df_params.execution_mode
        self._configured_select_cache: OrderedDict[tuple[str, ...], tuple[Any, Any]] = OrderedDict()
        self._configured_async_select_cache: OrderedDict[tuple[str, ...], tuple[Any, Any]] = OrderedDict()
        self._raw_sql_policy = raw_sql_policy
        self._policies = policies or GatewayPolicies()
        self._strict_filter_validation = strict_filter_validation
        self._allowed_filter_fields: set[str] = allowed_filter_fields or set()
        self._strategy.validate_requirements(
            config,
            require_datacube_request_validator=require_datacube_request_validator,
        )

        self._post_processor = PostProcessor(
            self._field_map,
            self._df_params,
            self._df_options,
            backend=self.backend,
            configured=self._configured,
            logger=self._logger,
        )

        self._configured_loader = self._build_configured_loader()

        self._auto_resolver = AutoReturnTypeResolver(
            config=self.config,
            resource=self.resource,
            strategy=self._strategy,
            field_map=self._field_map,
            async_sql_resource=self._async_sql_resource,
            df_params=self._df_params,
            configured=self._configured,
            build_configured_request=self._configured_loader._build_configured_request,
            get_configured_select=self._get_configured_select,
            get_configured_select_async=self._get_configured_select_async,
            configured_fieldnames=self._configured_loader._configured_fieldnames,
        )

    def _build_configured_loader(self) -> ConfiguredLoadService:
        return ConfiguredLoadService(
            strategy=self._strategy,
            table=self._table,
            config=self.config,
            resource=self.resource,
            field_map=self._field_map,
            sticky_filters=self._sticky_filters,
            exclude=self._exclude,
            df_params=self._df_params,
            post_processor=self._post_processor,
            async_sql_resource=self._async_sql_resource,
            get_configured_select=self._get_configured_select,
            get_configured_select_async=self._get_configured_select_async,
        )

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


    def _load_planner(self) -> LoadPlanner:
        return LoadPlanner(
            configured=self._configured,
            default_return_type=self._return_type,
            default_execution_mode=self._execution_mode,
            resolve_auto_return_type=self._resolve_auto_return_type,
            resolve_auto_return_type_async=self._resolve_auto_return_type_async,
        )

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
            self._configured_select_cache.move_to_end(cache_key)
            return cached
        assert isinstance(self.resource, SqlDatabaseResource)
        result = reflect_and_select(self.resource, self._table, db_columns)  # type: ignore[arg-type]
        self._configured_select_cache[cache_key] = result
        if len(self._configured_select_cache) > self._CACHE_MAXSIZE:
            self._configured_select_cache.popitem(last=False)
        return result

    async def _get_configured_select_async(
        self,
        resource: Any,
        db_columns: list[str] | None,
    ) -> tuple[Any, Any]:
        cache_key = self._configured_select_cache_key(db_columns)
        cached = self._configured_async_select_cache.get(cache_key)
        if cached is not None:
            self._configured_async_select_cache.move_to_end(cache_key)
            return cached
        result = await reflect_and_select_async(resource, self._table, db_columns)
        self._configured_async_select_cache[cache_key] = result
        if len(self._configured_async_select_cache) > self._CACHE_MAXSIZE:
            self._configured_async_select_cache.popitem(last=False)
        return result

    def _prepare_structured_loader_options(self, options: dict[str, Any]) -> dict[str, Any]:
        if self._configured:
            return options
        return self._strategy.prepare_structured_options(options, self._field_map, self._configured)

    def _resolve_auto_return_type(
        self,
        options: dict[str, Any],
    ) -> ResolvedReturnType:
        return self._auto_resolver.resolve(options)

    async def _resolve_auto_return_type_async(
        self,
        options: dict[str, Any],
    ) -> ResolvedReturnType:
        return await self._auto_resolver.resolve_async(options)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_backend(
        cls,
        backend: str,
        *,
        fs: fsspec.AbstractFileSystem | None = None,
        fs_factory: Any | None = None,
        **config_kwargs: Any,
    ) -> DataGateway:
        strategy = get_strategy(backend)
        config = strategy.build_config(**config_kwargs)
        return cls(config, fs=fs, fs_factory=fs_factory)

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

        backend, common = _factory.extract_config_common_options(cfg)
        strategy = get_strategy(backend)
        gateway_config = strategy.build_config_from_dict(cfg)
        return cls(gateway_config, **common.gateway_kwargs())

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
        await self._strategy.setup_async_context(self)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Load API — structured mode
    # ------------------------------------------------------------------

    def preview(self, statement: Any, model: Any, n: int = 5, npartitions: int = 1, **options: Any) -> pd.DataFrame:
        """Load a preview (first *n* rows) as a pandas DataFrame.

        Leverages lazy loading + ``safe_head`` so a distributed client is used
        when one is active.
        """
        frame = self.load(
            statement=statement,
            model=model,
            limit=n,
            return_type="dask",
            **options,
        )
        return safe_head(frame, n=n, npartitions=npartitions)

    async def apreview(self, statement: Any, model: Any, n: int = 5, npartitions: int = 1, **options: Any) -> pd.DataFrame:
        """Async version of :meth:`preview`."""
        frame = await self.aload(
            statement=statement,
            model=model,
            limit=n,
            return_type="dask",
            **options,
        )
        return await async_safe_head(frame, n=n, npartitions=npartitions)

    def _validate_raw_sql(self, options: dict[str, Any]) -> None:
        raw_sql = options.get("sql")
        if raw_sql is not None:
            allow_raw_sql = options.get("allow_raw_sql", False)
            if self._raw_sql_policy == "disabled":
                raise ValueError("Raw sql= execution is disabled by this DataGateway policy.")
            validate_raw_sql_statement(sql=raw_sql, allow_raw_sql=allow_raw_sql)

    def _validate_runtime_filters(self, loader_options: dict[str, Any]) -> None:
        if self._strict_filter_validation and self._configured:
            from boti_data.gateway.normalization import split_control_and_filters

            _ctrl, runtime_filters = split_control_and_filters(loader_options)
            for key in runtime_filters:
                field = key.split("__")[0]
                if field not in self._allowed_filter_fields:
                    raise ValueError(
                        f"Filter field '{field}' is not allowed. "
                        f"Allowed fields: {sorted(self._allowed_filter_fields)}"
                    )


    def _perform_load_sync(
        self,
        opts: dict[str, Any],
        *,
        request: GatewayLoadRequest | None = None,
        resolved_return_type: ReturnType,
        resolved_execution_mode: ResolvedExecutionMode,
        loader_return_type: Literal["pandas", "arrow", "dask"],
        loader_as_pandas: bool,
    ) -> pd.DataFrame | dd.DataFrame | pa.Table:
        if self._configured:
            return self._configured_loader.load(
                opts,
                return_type=resolved_return_type,
                execution_mode=resolved_execution_mode,
                loader_return_type=loader_return_type,
                loader_as_pandas=loader_as_pandas,
            )
        ctx = StructuredLoadContext(
            resource=self.resource,
            config=self.config,
            request=request,
            opts=opts,
            loader_return_type=loader_return_type,
            resolved_execution_mode=resolved_execution_mode,
            post_processor=self._post_processor,
            async_sql_resource=self._async_sql_resource,
        )
        return self._strategy.load_structured_sync(ctx)

    async def _perform_load_async(
        self,
        opts: dict[str, Any],
        *,
        request: GatewayLoadRequest | None = None,
        resolved_return_type: ReturnType,
        resolved_execution_mode: ResolvedExecutionMode,
        loader_return_type: Literal["pandas", "arrow", "dask"],
        loader_as_pandas: bool,
        timeout: float | None = None,
    ) -> pd.DataFrame | dd.DataFrame | pa.Table:
        if self._configured:
            coro = self._configured_loader.aload(
                opts,
                return_type=resolved_return_type,
                execution_mode=resolved_execution_mode,
                loader_return_type=loader_return_type,
                loader_as_pandas=loader_as_pandas,
            )
        else:
            ctx = StructuredLoadContext(
                resource=self.resource,
                config=self.config,
                request=request,
                opts=opts,
                loader_return_type=loader_return_type,
                resolved_execution_mode=resolved_execution_mode,
                timeout=timeout,
                post_processor=self._post_processor,
                async_sql_resource=self._async_sql_resource,
            )
            coro = self._strategy.load_structured_async(ctx)

        if timeout is not None:
            return await asyncio.wait_for(coro, timeout)
        return await coro

    def _prepare_load_options(
        self,
        options: dict[str, Any],
        plan: Any,
        controls: Any,
    ) -> dict[str, Any]:
        self._validate_raw_sql(options)
        if controls.diagnostics:
            self._post_processor.log_load_start(
                requested_return_type=plan.requested_return_type,
                resolved_return_type=plan.resolved_return_type,
                requested_execution_mode=plan.requested_execution_mode,
                resolved_execution_mode=plan.resolved_execution_mode,
                loader_return_type=plan.loader_return_type,
                persist=controls.persist,
            )
        loader_options = self._prepare_structured_loader_options(
            {**options, "as_pandas": plan.loader_as_pandas}
        )
        if controls.diagnostics:
            loader_options["diagnostics"] = True
        return loader_options

    def _finalize_load(
        self,
        df: FrameResult,
        plan: Any,
        controls: Any,
    ) -> FrameResult:
        return self._post_processor.finalize_load_result(
            df,
            plan.strategy,
            controls.persist,
            controls.resilient,
            controls.dry_run,
            controls.diagnostics,
            plan.started,
        )

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
        control, _ = split_control_and_filters(options)
        request = GatewayLoadRequest.model_validate(control)
        plan = self._load_planner().plan(options, request=request)
        controls = plan.controls
        loader_options = self._prepare_load_options(options, plan, controls)

        loader_options = _series_filters.resolve_series_filters(loader_options)
        self._validate_runtime_filters(loader_options)
        in_chunk_size, in_chunk_concurrency = self._resolve_in_chunk_controls(
            loader_options,
            strategy=controls.in_chunk_strategy,
            execution_mode=plan.resolved_execution_mode,
            in_chunk_size_raw=controls.in_chunk_size_raw,
            in_chunk_concurrency_raw=controls.in_chunk_concurrency_raw,
        )

        def _execute_sync(**opts: Any) -> pd.DataFrame | dd.DataFrame | pa.Table:
            return self._perform_load_sync(
                opts,
                request=request,
                resolved_return_type=plan.resolved_return_type,
                resolved_execution_mode=plan.resolved_execution_mode,
                loader_return_type=plan.loader_return_type,
                loader_as_pandas=plan.loader_as_pandas,
            )

        df = self._chunked_in_load_sync(
            _execute_sync,
            in_chunk_size,
            loader_options,
            return_type=plan.resolved_return_type,
            max_concurrency=in_chunk_concurrency,
        )
        return self._finalize_load(df, plan, controls)

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
        control, _ = split_control_and_filters(options)
        request = GatewayLoadRequest.model_validate(control)
        plan = await self._load_planner().aplan(options, request=request)
        controls = plan.controls
        loader_options = self._prepare_load_options(options, plan, controls)

        # Resolve any Series values before chunked dispatch so the chunker
        # sees plain lists (which it already knows how to split).
        loader_options = await _series_filters.resolve_series_filters_async(
            loader_options
        )
        self._validate_runtime_filters(loader_options)
        in_chunk_size, in_chunk_concurrency = self._resolve_in_chunk_controls(
            loader_options,
            strategy=controls.in_chunk_strategy,
            execution_mode=plan.resolved_execution_mode,
            in_chunk_size_raw=controls.in_chunk_size_raw,
            in_chunk_concurrency_raw=controls.in_chunk_concurrency_raw,
        )

        async def _execute(**opts: Any) -> pd.DataFrame | dd.DataFrame | pa.Table:
            return await self._perform_load_async(
                opts,
                request=request,
                resolved_return_type=plan.resolved_return_type,
                resolved_execution_mode=plan.resolved_execution_mode,
                loader_return_type=plan.loader_return_type,
                loader_as_pandas=plan.loader_as_pandas,
                timeout=controls.timeout,
            )

        df = await self._chunked_in_load(
            _execute,
            in_chunk_size,
            loader_options,
            return_type=plan.resolved_return_type,
            max_concurrency=in_chunk_concurrency,
        )
        return self._finalize_load(df, plan, controls)

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
        return await ChunkedLoadExecutor.aload(
            execute_fn,
            chunk_size,
            options,
            return_type=return_type,
            max_concurrency=max_concurrency,
        )

    def _chunked_in_load_sync(
        self,
        execute_fn: Any,
        chunk_size: int,
        options: dict[str, Any],
        *,
        return_type: ReturnType,
        max_concurrency: int | None = None,
    ) -> FrameResult:
        return ChunkedLoadExecutor.load(
            execute_fn,
            chunk_size,
            options,
            return_type=return_type,
            max_concurrency=max_concurrency,
            logger=self._logger,
        )

    def _resolve_in_chunk_controls(
        self,
        options: dict[str, Any],
        *,
        strategy: str,
        execution_mode: str | None = None,
        in_chunk_size_raw: Any,
        in_chunk_concurrency_raw: Any,
    ) -> tuple[int, int | None]:
        planner = InChunkPlanner(
            strategy=self._strategy,
            policy_provider=lambda: self._policies.in_chunk_policy,
        )
        return planner.resolve_controls(
            options,
            strategy=strategy,
            execution_mode=execution_mode,
            in_chunk_size_raw=in_chunk_size_raw,
            in_chunk_concurrency_raw=in_chunk_concurrency_raw,
        )

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
        key_frame = _series_filters.series_to_dask_key_frame(join_series, column_name=on)
        if on in frame.columns:
            key_frame = key_frame.astype({on: frame.dtypes[on]})
        joined = frame.merge(key_frame, how="inner", on=on)
        return joined[list(frame.columns)]

    @staticmethod
    def _has_any_rows(df: FrameResult) -> bool:
        """Return ``True`` if *df* contains at least one row."""
        try:
            return strategy_for_frame(df).has_any_rows(df)
        except Exception:
            _log.debug("Failed to check has_any_rows", exc_info=True)
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

    async def aclose(self) -> None:
        await self._strategy.teardown_async_context(self)
        self.close()




