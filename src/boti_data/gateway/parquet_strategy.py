"""Strategy for the ``parquet`` backend.

Split out of _backend_strategies.py purely for line-count headroom.
Registered back into that module's registry at the bottom of
_backend_strategies.py to avoid a circular import (this module needs
``BackendStrategy``/``StructuredLoadContext``/``ConfiguredLoadContext``
already defined there).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import fsspec

from boti_data.parquet.resource import ParquetDataConfig, ParquetDataResource

from . import _factory, _payloads
from ._backend_strategies import BackendStrategy, ConfiguredLoadContext, StructuredLoadContext
from .frame_strategies import FrameResult
from .loaders import load_parquet
from .requests import BackendConfig, BackendName, BackendResource, ParquetLoadRequest
from .sql_size_estimation import _AUTO_EAGER_MAX_ROWS

_log = logging.getLogger(__name__)

_AUTO_EAGER_MAX_FILES = 4


def _resolve_parquet_scan_summary(
    resource: ParquetDataResource,
) -> tuple[int | None, int | None]:
    return resource.scan_summary(max_files=_AUTO_EAGER_MAX_FILES)


def _safe_resolve_parquet_scan_summary(
    resource: ParquetDataResource,
) -> tuple[int | None, int | None] | None:
    try:
        return _resolve_parquet_scan_summary(resource)
    except Exception:
        _log.debug("Failed to resolve parquet scan summary", exc_info=True)
        return None


def _estimate_via_scan_summary(
    limit: int | None,
    resource: Any,
) -> tuple[int | None, int | None] | None:
    if isinstance(limit, int) and limit <= _AUTO_EAGER_MAX_ROWS:
        return limit, None
    if not isinstance(resource, ParquetDataResource):
        return None
    return _safe_resolve_parquet_scan_summary(resource)


class ParquetStrategy(BackendStrategy):
    """Strategy for the ``parquet`` backend."""

    @property
    def name(self) -> BackendName:
        return "parquet"

    # -- Config construction -------------------------------------------------

    def build_config(self, **kwargs: Any) -> ParquetDataConfig:
        return ParquetDataConfig(**kwargs)

    def build_config_from_dict(self, cfg: dict[str, Any]) -> ParquetDataConfig:
        cfg = dict(cfg)
        storage_path = cfg.pop("storage_path", None)
        if storage_path is not None and "parquet_storage_path" not in cfg:
            cfg["parquet_storage_path"] = storage_path
        if _factory.should_default_partition_on(cfg):
            cfg["partition_on"] = ["partition_date"]
        return ParquetDataConfig(**cfg)

    # -- Resource construction -----------------------------------------------

    def build_resource(
        self,
        config: BackendConfig,
        *,
        fs: fsspec.AbstractFileSystem | None = None,
        fs_factory: Any | None = None,
    ) -> tuple[BackendName, BackendResource]:
        assert isinstance(config, ParquetDataConfig)
        return "parquet", ParquetDataResource(config, fs=fs, fs_factory=fs_factory)

    # -- Structured mode loads -----------------------------------------------

    @staticmethod
    def _build_structured_parquet_request(ctx: StructuredLoadContext) -> ParquetLoadRequest:
        # Use ctx.opts (which still carries bare runtime filter kwargs) rather
        # than ctx.request: GatewayLoadRequest forbids extra fields, so bare
        # kwargs never land on request.filters and would be silently dropped.
        payload_source: Any = ctx.opts if ctx.opts is not None else ctx.request
        return ParquetLoadRequest.model_validate(
            _payloads.structured_parquet_request_payload(
                payload_source,
                return_type=ctx.loader_return_type,
            )
        )

    def load_structured_sync(self, ctx: StructuredLoadContext) -> FrameResult:
        request = self._build_structured_parquet_request(ctx)
        return load_parquet(ctx.resource, request)

    async def load_structured_async(self, ctx: StructuredLoadContext) -> FrameResult:
        request = self._build_structured_parquet_request(ctx)
        return await asyncio.to_thread(load_parquet, ctx.resource, request)

    # -- Configured mode loads -----------------------------------------------

    @staticmethod
    def _build_configured_parquet_request(ctx: ConfiguredLoadContext) -> ParquetLoadRequest:
        return ParquetLoadRequest.model_construct(
            filters=ctx.combined_filters,
            raw_filters=ctx.control.raw_filters,
            limit=ctx.control.limit,
            columns=(list(ctx.configured_fieldnames) if ctx.configured_fieldnames else None),
            as_pandas=ctx.loader_as_pandas,
            diagnostics=ctx.control.diagnostics,
            return_type=ctx.loader_return_type,
        )

    @staticmethod
    def _finalize_configured_parquet_result(
        ctx: ConfiguredLoadContext, df: FrameResult
    ) -> FrameResult:
        return ctx.post_processor.finalize_configured_result(
            df,
            return_type=ctx.return_type,
            apply_field_map=False,
            fieldnames=ctx.configured_fieldnames,
        )

    def load_configured_sync(self, ctx: ConfiguredLoadContext) -> FrameResult:
        assert isinstance(ctx.resource, ParquetDataResource)
        df = load_parquet(ctx.resource, self._build_configured_parquet_request(ctx))
        return self._finalize_configured_parquet_result(ctx, df)

    async def load_configured_async(self, ctx: ConfiguredLoadContext) -> FrameResult:
        # load_configured_sync() is pure blocking parquet I/O; run the whole
        # thing off-thread rather than re-implementing it here.
        return await asyncio.to_thread(self.load_configured_sync, ctx)

    # -- Auto return type ----------------------------------------------------

    def estimate_result_size(
        self,
        ctx: StructuredLoadContext | ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        if isinstance(ctx, StructuredLoadContext):
            return self._estimate_structured(ctx)
        return self._estimate_configured(ctx)

    @staticmethod
    def _estimate_structured(
        ctx: StructuredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        return _estimate_via_scan_summary(ctx.opts.get("limit"), ctx.resource)

    @staticmethod
    def _estimate_configured(
        ctx: ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        return _estimate_via_scan_summary(ctx.control.limit, ctx.resource)
