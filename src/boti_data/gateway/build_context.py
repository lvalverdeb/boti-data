"""Shared wiring passed to DataGateway's internal collaborators at construction.

Split out to fix `too-many-params` on ConfiguredLoadService/LoadExecutor/
AutoReturnTypeResolver's constructors: each was individually threading the
same handful of gateway-wide values (config, resource, strategy, field_map,
df_params, configured, async_sql_resource) as separate parameters.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from boti_data.db import AsyncSqlDatabaseResource
from boti_data.field_map import FieldMap

from ._backend_strategies import BackendStrategy
from .post_process import PostProcessor
from .requests import BackendConfig, BackendResource, DataFrameParams


@dataclass(frozen=True)
class GatewayBuildContext:
    """Gateway-wide values needed by more than one internal collaborator.

    Built once in DataGateway.__init__ and passed to each collaborator's
    constructor, which copies out whichever fields it needs into its own
    attributes (including `async_sql_resource`, which individual collaborators
    continue to update post-construction via their own `update_async_resource`
    — this frozen context only supplies the initial value).
    """

    config: BackendConfig
    resource: BackendResource | None
    strategy: BackendStrategy
    field_map: FieldMap
    df_params: DataFrameParams
    configured: bool
    post_processor: PostProcessor
    async_sql_resource: AsyncSqlDatabaseResource | None = None


@dataclass(frozen=True)
class ConfiguredSelectAccessors:
    """The paired sync/async configured-select callbacks, always built and
    passed together to ConfiguredLoadService and AutoReturnTypeResolver."""

    get_configured_select: Callable[[list[str] | None], tuple[Any, Any]] | None = None
    get_configured_select_async: (
        Callable[[Any, list[str] | None], Awaitable[tuple[Any, Any]]] | None
    ) = None


@dataclass(frozen=True)
class LoadValidationPolicy:
    """Raw-SQL and runtime-filter validation policy for LoadExecutor."""

    raw_sql_policy: str | None
    strict_filter_validation: bool
    allowed_filter_fields: set[str]


@dataclass(frozen=True)
class LoadExecutorCallbacks:
    """DataGateway-bound resolver callbacks LoadExecutor calls back into."""

    resolve_auto_return_type: Callable[[dict[str, Any]], Any]
    resolve_auto_return_type_async: Callable[[dict[str, Any]], Awaitable[Any]]
    resolve_in_chunk_controls: Callable[..., tuple[int, int | None]]
