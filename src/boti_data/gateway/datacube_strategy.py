"""Strategy for the ``datacube`` backend.

Split out of _backend_strategies.py purely for line-count headroom.
Registered back into that module's registry at the bottom of
_backend_strategies.py to avoid a circular import (this module needs
``BackendStrategy``/``StructuredLoadContext``/``ConfiguredLoadContext``
already defined there).
"""

from __future__ import annotations

from typing import Any

import fsspec

from boti_data.datacube import DatacubeConfig, DatacubeResource

from . import _payloads
from ._backend_strategies import BackendStrategy, ConfiguredLoadContext, StructuredLoadContext
from .frame_strategies import FrameResult, get_frame_strategy
from .requests import BackendConfig, BackendName, BackendResource


class DatacubeStrategy(BackendStrategy):
    """Strategy for the ``datacube`` backend."""

    @property
    def name(self) -> BackendName:
        return "datacube"

    # -- Config construction -------------------------------------------------

    def build_config(self, **kwargs: Any) -> DatacubeConfig:
        return DatacubeConfig(**kwargs)

    def build_config_from_dict(self, cfg: dict[str, Any]) -> DatacubeConfig:
        return DatacubeConfig(**cfg)

    # -- Resource construction -----------------------------------------------

    def build_resource(
        self,
        config: BackendConfig,
        *,
        fs: fsspec.AbstractFileSystem | None = None,
        fs_factory: Any | None = None,
    ) -> tuple[BackendName, BackendResource]:
        assert isinstance(config, DatacubeConfig)
        return "datacube", DatacubeResource(config)

    # -- Config validation ---------------------------------------------------

    def validate_requirements(self, config: BackendConfig, **flags: Any) -> None:
        required = flags.get("require_datacube_request_validator", False)
        if not required:
            return
        if not isinstance(config, DatacubeConfig):
            raise ValueError("require_datacube_request_validator=True requires DatacubeConfig.")
        contract = config.contract
        if contract is None or contract.request_validator is None:
            raise ValueError(
                "require_datacube_request_validator=True requires DatacubeContract "
                "with a request_validator."
            )

    # -- Structured mode loads -----------------------------------------------

    # Not a copy-pasted twin: sync calls the resource's native ctx.resource.load(),
    # async calls the resource's native ctx.resource.aload() — a genuinely-async
    # collaborator method, not a thread-wrap; request-building is already
    # shared via _payloads.datacube_request_from_options().
    # spaghetti-ignore[sync-async-duplication]
    def load_structured_sync(self, ctx: StructuredLoadContext) -> FrameResult:
        assert isinstance(ctx.resource, DatacubeResource) or hasattr(ctx.resource, "load")
        # Use ctx.opts (includes runtime filter values) rather than ctx.request
        # (which only has control keys — runtime filters are excluded from the
        # GatewayLoadRequest model by design).
        return ctx.resource.load(_payloads.datacube_request_from_options(ctx.opts))

    async def load_structured_async(self, ctx: StructuredLoadContext) -> FrameResult:
        assert isinstance(ctx.resource, DatacubeResource) or hasattr(ctx.resource, "aload")
        return await ctx.resource.aload(_payloads.datacube_request_from_options(ctx.opts))

    # -- Configured mode loads -----------------------------------------------

    @staticmethod
    def _build_configured_datacube_request(ctx: ConfiguredLoadContext) -> Any:
        from boti_data.datacube.contract import DatacubeRequest

        return DatacubeRequest(
            filters=ctx.combined_filters,
            cube=ctx.control.cube,
        )

    # Not a copy-pasted twin: _build_configured_datacube_request() is already
    # shared; sync calls resource.load, async calls the native resource.aload —
    # nothing left to hoist beyond that irreducible call.
    # spaghetti-ignore[sync-async-duplication]
    def load_configured_sync(self, ctx: ConfiguredLoadContext) -> FrameResult:
        request = self._build_configured_datacube_request(ctx)
        df = ctx.resource.load(request)
        return get_frame_strategy(ctx.return_type).normalize(df)

    async def load_configured_async(self, ctx: ConfiguredLoadContext) -> FrameResult:
        request = self._build_configured_datacube_request(ctx)
        df = await ctx.resource.aload(request)
        return get_frame_strategy(ctx.return_type).normalize(df)

    # -- Chunking ------------------------------------------------------------

    def supports_chunking(self) -> bool:
        return False
