"""DataGateway.__init__ construction logic, split out for line-count headroom.

DataGateway.__init__ wires together roughly a dozen internal collaborators
(strategy, resource, post-processor, configured-load service, auto return-type
resolver, semi-join service, load executor). None of that wiring depends on
DataGateway's own methods being callable yet except by *reference* (the
callbacks below are bound methods looked up lazily, only invoked long after
construction finishes), so it moves here as free functions taking the
not-yet-fully-constructed gateway instance explicitly and setting its
attributes directly — mirroring the established pattern of async orchestration
functions elsewhere in this codebase that take the owning instance as their
first parameter instead of being methods themselves.

``GatewayInitOptions`` bundles DataGateway.__init__'s many keyword arguments
into one object purely so these internal helpers stay under the
too-many-params limit; DataGateway.__init__ itself keeps its full public
keyword-argument signature (a deliberate, previously-reviewed exception —
public API stability wins over the params-count check there) and constructs
the options bundle itself before calling build_gateway_state().
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import fsspec

from boti_data.field_map import FieldMap

from . import _factory
from ._backend_strategies import for_config
from ._backend_strategies import get as get_strategy
from .build_context import (
    ConfiguredSelectAccessors,
    GatewayBuildContext,
    LoadExecutorCallbacks,
    LoadValidationPolicy,
)
from .configured_load import ConfiguredLoadService
from .load_execution import LoadExecutor
from .policies import GatewayPolicies
from .post_process import PostProcessor, ResultShapingConfig
from .requests import BackendConfig, DataFrameOptions, DataFrameParams
from .return_type import AutoReturnTypeResolver
from .select_cache import ConfiguredSelectCache
from .semi_join import SemiJoinService

if TYPE_CHECKING:
    from .core import DataGateway


@dataclass(frozen=True)
class GatewayInitOptions:
    """Bundles DataGateway.__init__'s keyword arguments for its internal builders."""

    field_map: dict[str, str] | None = None
    table: str | None = None
    sticky_filters: dict[str, Any] | None = None
    exclude: bool = False
    df_params: DataFrameParams | None = None
    df_options: DataFrameOptions | None = None
    fs: fsspec.AbstractFileSystem | None = None
    fs_factory: Any | None = None
    raw_sql_policy: str | None = None
    policies: GatewayPolicies | None = None
    strict_filter_validation: bool = False
    allowed_filter_fields: set[str] | None = None
    require_datacube_request_validator: bool = False


def _init_resource(
    gateway: DataGateway, config: BackendConfig, options: GatewayInitOptions
) -> BackendConfig:
    config = _factory.coerce_backend_config(config)
    gateway.config = config
    gateway._strategy = for_config(config)
    gateway.backend, gateway.resource = gateway._strategy.build_resource(
        config,
        fs=options.fs,
        fs_factory=options.fs_factory,
    )
    gateway._async_sql_resource = None
    return config


def _init_configured_mode_state(
    gateway: DataGateway, config: BackendConfig, options: GatewayInitOptions
) -> tuple[FieldMap, DataFrameParams, bool]:
    gateway._table = options.table
    configured = options.table is not None
    field_map_obj = FieldMap(options.field_map) if options.field_map else FieldMap({})
    df_params = options.df_params or DataFrameParams()
    df_options = options.df_options or DataFrameOptions()
    gateway._return_type = df_params.return_type
    gateway._execution_mode = df_params.execution_mode
    gateway._select_cache = ConfiguredSelectCache()
    gateway._policies = options.policies or GatewayPolicies()
    gateway._strategy.validate_requirements(
        config,
        require_datacube_request_validator=options.require_datacube_request_validator,
    )
    gateway._post_processor = PostProcessor(
        ResultShapingConfig(field_map_obj, df_params, df_options),
        backend=gateway.backend,
        configured=configured,
        logger=gateway.logger,
    )
    return field_map_obj, df_params, configured


def _build_select_accessors(gateway: DataGateway) -> ConfiguredSelectAccessors:
    get_configured_select = functools.partial(
        gateway._select_cache.get, gateway.resource, gateway._table
    )
    get_configured_select_async = functools.partial(gateway._select_cache.get_async, gateway._table)
    return ConfiguredSelectAccessors(
        get_configured_select=get_configured_select,
        get_configured_select_async=get_configured_select_async,
    )


def _build_context(
    gateway: DataGateway, field_map_obj: FieldMap, df_params: DataFrameParams, *, configured: bool
) -> GatewayBuildContext:
    return GatewayBuildContext(
        config=gateway.config,
        resource=gateway.resource,
        strategy=gateway._strategy,
        field_map=field_map_obj,
        df_params=df_params,
        configured=configured,
        post_processor=gateway._post_processor,
        async_sql_resource=gateway._async_sql_resource,
    )


def _build_services(
    gateway: DataGateway,
    ctx: GatewayBuildContext,
    select_accessors: ConfiguredSelectAccessors,
    df_params: DataFrameParams,
    options: GatewayInitOptions,
) -> None:
    gateway._configured_loader = ConfiguredLoadService(
        ctx,
        table=gateway._table,
        sticky_filters=dict(options.sticky_filters or {}),
        exclude=options.exclude,
        select_accessors=select_accessors,
    )
    gateway._auto_resolver = AutoReturnTypeResolver(
        ctx,
        build_configured_request=gateway._configured_loader._build_configured_request,
        select_accessors=select_accessors,
        configured_fieldnames=gateway._configured_loader._configured_fieldnames,
    )
    gateway._semi_join_service = SemiJoinService(
        configured=ctx.configured,
        df_params=df_params,
        default_return_type=gateway._return_type,
        load=gateway.load,
    )
    gateway._load_executor = LoadExecutor(
        ctx,
        configured_loader=gateway._configured_loader,
        validation_policy=LoadValidationPolicy(
            raw_sql_policy=options.raw_sql_policy,
            strict_filter_validation=options.strict_filter_validation,
            allowed_filter_fields=options.allowed_filter_fields or set(),
        ),
        callbacks=LoadExecutorCallbacks(
            resolve_auto_return_type=gateway._resolve_auto_return_type,
            resolve_auto_return_type_async=gateway._resolve_auto_return_type_async,
            resolve_in_chunk_controls=gateway._resolve_in_chunk_controls,
        ),
    )


def build_gateway_state(
    gateway: DataGateway, config: BackendConfig, options: GatewayInitOptions
) -> None:
    """Set every attribute DataGateway.__init__ needs, in place, on *gateway*."""
    config = _init_resource(gateway, config, options)
    field_map_obj, df_params, configured = _init_configured_mode_state(gateway, config, options)
    select_accessors = _build_select_accessors(gateway)
    ctx = _build_context(gateway, field_map_obj, df_params, configured=configured)
    _build_services(gateway, ctx, select_accessors, df_params, options)


def build_gateway_from_backend(
    cls: type[DataGateway],
    backend: str,
    *,
    fs: fsspec.AbstractFileSystem | None,
    fs_factory: Any | None,
    config_kwargs: dict[str, Any],
) -> DataGateway:
    strategy = get_strategy(backend)
    config = strategy.build_config(**config_kwargs)
    return cls(config, fs=fs, fs_factory=fs_factory)


def build_gateway_from_config(
    cls: type[DataGateway],
    config: dict[str, Any],
    overrides: dict[str, Any],
) -> DataGateway:
    cfg = dict(config)
    # Wrapper classes (e.g. ParquetReader -> DataHelper) always forward
    # fs=None/fs_factory=None when the caller didn't explicitly pass them,
    # rather than omitting the keys. A blind cfg.update(overrides) would let
    # that incidental None clobber a real fs embedded directly in the config
    # mapping, so only non-None overrides are allowed to win.
    cfg.update({key: value for key, value in overrides.items() if value is not None})

    backend, common = _factory.extract_config_common_options(cfg)
    strategy = get_strategy(backend)
    gateway_config = strategy.build_config_from_dict(cfg)
    return cls(gateway_config, **common.gateway_kwargs())
