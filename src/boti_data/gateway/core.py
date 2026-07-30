"""
Dask-first gateway over existing boti_data backend resources.
"""

from __future__ import annotations

from typing import Any

import dask.dataframe as dd
import fsspec
import pandas as pd
import polars as pl
from boti.core.lifecycle import LifecycleCore
from boti.core.lifecycle_pickle import PicklableLifecycleCoreMixin
from boti_dask import async_safe_head, safe_head

from . import core_describe, core_load
from ._gateway_init import (
    GatewayInitOptions,
    build_gateway_from_backend,
    build_gateway_from_config,
    build_gateway_state,
)
from .frame_strategies import FrameResult
from .normalization import prepare_period_filters
from .policies import GatewayPolicies
from .requests import (
    BackendConfig,
    DataFrameOptions,
    DataFrameParams,
    ExecutionMode,
    ResolvedReturnType,
    ReturnType,
    TableDescription,
)


class DataGateway(PicklableLifecycleCoreMixin, LifecycleCore):
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
        raw_sql_policy: str | None = None,
        policies: GatewayPolicies | None = None,
        strict_filter_validation: bool = False,
        allowed_filter_fields: set[str] | None = None,
        require_datacube_request_validator: bool = False,
    ) -> None:
        build_gateway_state(
            self,
            config,
            GatewayInitOptions(
                field_map=field_map,
                table=table,
                sticky_filters=sticky_filters,
                exclude=exclude,
                df_params=df_params,
                df_options=df_options,
                fs=fs,
                fs_factory=fs_factory,
                raw_sql_policy=raw_sql_policy,
                policies=policies,
                strict_filter_validation=strict_filter_validation,
                allowed_filter_fields=allowed_filter_fields,
                require_datacube_request_validator=require_datacube_request_validator,
            ),
        )
        # self.resource (set above) is what logger reads through to, so
        # LifecycleCore's GC finalizer captures a real logger where available
        # instead of defaulting to None.
        super().__init__()

    @property
    def default_return_type(self) -> ReturnType:
        return self._return_type

    @property
    def default_execution_mode(self) -> ExecutionMode:
        return self._execution_mode

    @property
    def logger(self) -> Any | None:
        """LifecycleCore's error-logging and GC leak-warning read this.

        Overrides LifecycleCore's plain `logger = None` class default to
        reuse whichever backend resource's logger is already configured,
        rather than defaulting to no logger at all.
        """
        if self._async_sql_resource is not None:
            return getattr(self._async_sql_resource, "logger", None)
        if self.resource is not None:
            return getattr(self.resource, "logger", None)
        return None

    # Not a copy-pasted twin: pure pass-through to self._auto_resolver.resolve()/
    # resolve_async(), which already share the real decision logic.
    # spaghetti-ignore[sync-async-duplication]: see above
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

    # --- Constructors ---

    @classmethod
    def from_backend(
        cls,
        backend: str,
        *,
        fs: fsspec.AbstractFileSystem | None = None,
        fs_factory: Any | None = None,
        **config_kwargs: Any,
    ) -> DataGateway:
        return build_gateway_from_backend(
            cls, backend, fs=fs, fs_factory=fs_factory, config_kwargs=config_kwargs
        )

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
        return build_gateway_from_config(cls, config, overrides)

    # --- Lifecycle ---

    async def __aenter__(self) -> DataGateway:
        await super().__aenter__()
        await self._strategy.setup_async_context(self)
        return self

    def _cleanup(self) -> None:
        if self.resource is not None:
            self.resource.close()

    async def _acleanup(self) -> None:
        await self._strategy.teardown_async_context(self)
        self._cleanup()

    # __getstate__/__setstate__ come from PicklableLifecycleCoreMixin; pickle-
    # security gating is enforced by the wrapped `resource` (a ManagedResource),
    # pickled recursively as part of __dict__ — no gate needed here too.

    # --- Load API — structured mode ---

    # Not a copy-pasted twin: async_safe_head() is already a thin
    # asyncio.to_thread() wrapper around safe_head(); the other half
    # (self.load vs await self.aload) is genuinely-different async I/O that
    # the codebase deliberately keeps separate.
    # spaghetti-ignore[sync-async-duplication]: see above
    def preview(
        self, statement: Any, model: Any, n: int = 5, npartitions: int = 1, **options: Any
    ) -> pd.DataFrame:
        """Load a preview (first *n* rows) as a pandas DataFrame.

        Leverages lazy loading + ``safe_head`` so a distributed client is used
        when one is active.
        """
        frame = self.load(statement=statement, model=model, limit=n, return_type="dask", **options)
        return safe_head(frame, n=n, npartitions=npartitions)

    async def apreview(
        self, statement: Any, model: Any, n: int = 5, npartitions: int = 1, **options: Any
    ) -> pd.DataFrame:
        """Async version of :meth:`preview`."""
        frame = await self.aload(
            statement=statement, model=model, limit=n, return_type="dask", **options
        )
        return await async_safe_head(frame, n=n, npartitions=npartitions)

    def describe(
        self, table: str | None = None, *, row_count_limit: int = 10_000
    ) -> TableDescription:
        """Cheaply inspect a table's schema and an approximate row count.

        Runs a bounded introspection query — reflecting column dtypes without
        loading any rows, then counting up to *row_count_limit* rows — instead
        of requiring callers to know to self-impose ``limit=`` on a full
        :meth:`load` before exploring an unfamiliar table.

        Args:
            table: DB table name. Defaults to the table configured at
                construction time (configured mode).
            row_count_limit: Row count is exact up to this many rows; beyond
                it, :attr:`TableDescription.row_count_is_exact` is ``False``
                and :attr:`TableDescription.row_count` reports this cap.
        """
        return core_describe.describe(self, table, row_count_limit=row_count_limit)

    async def adescribe(
        self, table: str | None = None, *, row_count_limit: int = 10_000
    ) -> TableDescription:
        """Async variant of :meth:`describe`."""
        return await core_describe.describe_async(self, table, row_count_limit=row_count_limit)

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
        return core_load.load_sync(self, options)

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
        return await core_load.load_async(self, options)

    def load_period(
        self,
        dt_field: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> FrameResult:
        """Load rows within a date range.

        ``dt_field`` is always expressed as a **semantic name**.  When a
        ``field_map`` is configured the gateway translates it to the DB column
        name automatically during the load — no manual mapping needed here.

        Uses direct ``__exact`` / ``__gte`` + ``__lte`` comparison so the
        filters work correctly for DATE, TIMESTAMP, and ISO-text columns across
        all supported databases.  If you specifically need date-part extraction
        from a full DATETIME column, pass ``field__date__gte`` / ``__lte``
        filters directly instead.

        Args:
            dt_field: Semantic column name of the datetime/date field.
            start: ISO-8601 start date (inclusive).
            end: ISO-8601 end date (inclusive).
            **kwargs: Forwarded to :meth:`load` as control or filter kwargs.
        """
        return self.load(**prepare_period_filters(dt_field, start, end, **kwargs))

    async def aload_period(
        self,
        dt_field: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> FrameResult:
        """Async variant of :meth:`load_period`."""
        return await self.aload(**prepare_period_filters(dt_field, start, end, **kwargs))

    # Kept as real methods (not module-function aliases): bound-method
    # pickling for distributed use and test monkeypatching of
    # `self._chunked_in_load` both need `self` bound under this exact name.
    # spaghetti-ignore[sync-async-duplication]: see above
    async def _chunked_in_load(
        self,
        execute_fn: Any,
        chunk_size: int,
        options: dict[str, Any],
        *,
        return_type: ReturnType,
        max_concurrency: int | None = None,
    ) -> FrameResult:
        kwargs: dict[str, Any] = {"return_type": return_type, "max_concurrency": max_concurrency}
        return await core_load.chunked_in_load(self, execute_fn, chunk_size, options, **kwargs)

    def _chunked_in_load_sync(
        self,
        execute_fn: Any,
        chunk_size: int,
        options: dict[str, Any],
        *,
        return_type: ReturnType,
        max_concurrency: int | None = None,
    ) -> FrameResult:
        kwargs: dict[str, Any] = {"return_type": return_type, "max_concurrency": max_concurrency}
        return core_load.chunked_in_load_sync(self, execute_fn, chunk_size, options, **kwargs)

    def _resolve_in_chunk_controls(
        self,
        options: dict[str, Any],
        *,
        strategy: str,
        execution_mode: str | None = None,
        in_chunk_size_raw: Any,
        in_chunk_concurrency_raw: Any,
    ) -> tuple[int, int | None]:
        kwargs: dict[str, Any] = {
            "strategy": strategy,
            "execution_mode": execution_mode,
            "in_chunk_size_raw": in_chunk_size_raw,
            "in_chunk_concurrency_raw": in_chunk_concurrency_raw,
        }
        return core_load.resolve_in_chunk_controls(self, options, **kwargs)

    @staticmethod
    def _has_any_rows(df: FrameResult) -> bool:
        return core_load.has_any_rows(df)

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
        return core_load.semi_join_sync(self, join_series, on, kwargs)

    async def asemi_join(
        self,
        join_series: pd.Series | dd.Series | pl.Series,
        on: str,
        **kwargs: Any,
    ) -> FrameResult:
        """Async variant of :meth:`semi_join`."""
        return await core_load.semi_join_async(self, join_series, on, kwargs)
