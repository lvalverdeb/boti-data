"""Backend strategy registry and base interface.

Each backend (sqlalchemy, parquet, datacube) is encapsulated in a class
that implements :class:`BackendStrategy`.  Adding a new backend means
creating a new strategy class, implementing the abstract methods, and
registering it — no changes to dispatch chains in *core.py*,
*configured_load.py*, or *return_type.py*.

The concrete strategy classes live in sibling modules (sql_strategy.py,
parquet_strategy.py, datacube_strategy.py), split out purely for line-count
headroom, and are imported and registered at the bottom of this file — after
``BackendStrategy``/``StructuredLoadContext``/``ConfiguredLoadContext`` are
already defined, so importing them here creates no circular-import problem.

``load_sql``/``load_sql_partitioned``/``AsyncSqlDatabaseResource`` stay
imported directly into *this* module's namespace (rather than only inside
sql_strategy.py) because tests monkeypatch those three names on
``boti_data.gateway._backend_strategies`` directly; sql_strategy.py calls
them back through this module object so the patched value is always picked
up at call time.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import fsspec

from boti_data.datacube import DatacubeConfig

# spaghetti-ignore[unused-import]: monkeypatch target, see module docstring
from boti_data.db import AsyncSqlDatabaseResource, SqlDatabaseConfig  # noqa: F401
from boti_data.field_map import FieldMap
from boti_data.parquet.resource import ParquetDataConfig

from .frame_strategies import FrameResult
from .load_request import GatewayLoadRequest

# spaghetti-ignore[unused-import]: monkeypatch target, see module docstring
from .loaders import load_sql, load_sql_partitioned  # noqa: F401
from .planning import InChunkPolicy
from .post_process import PostProcessor
from .requests import (
    BackendConfig,
    BackendName,
    BackendResource,
    ResolvedExecutionMode,
    ReturnType,
)

_log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .core import DataGateway


# ---------------------------------------------------------------------------
# Context dataclasses — single-parameter wrappers against parameter bloat
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredLoadContext:
    """Everything a strategy needs to execute a structured-mode load."""

    resource: BackendResource | None
    config: BackendConfig
    opts: dict[str, Any]
    loader_return_type: Literal["pandas", "arrow", "dask"]
    resolved_execution_mode: ResolvedExecutionMode | None = None
    timeout: float | None = None
    post_processor: PostProcessor | None = None
    async_sql_resource: Any | None = None
    request: GatewayLoadRequest | None = None


@dataclass(frozen=True)
class ConfiguredLoadContext:
    """Everything a strategy needs to execute a configured-mode load."""

    resource: BackendResource | None
    config: BackendConfig
    control: GatewayLoadRequest
    combined_filters: dict[str, Any]
    db_filters: dict[str, Any]
    db_columns: list[str] | None
    configured_fieldnames: tuple[str, ...] | None
    field_map: FieldMap
    exclude: bool
    sticky_filters: dict[str, Any]
    return_type: ReturnType
    loader_return_type: Literal["pandas", "arrow", "dask"]
    loader_as_pandas: bool
    execution_mode: ResolvedExecutionMode
    post_processor: PostProcessor
    get_configured_select: Callable[[list[str] | None], tuple[Any, Any]] | None = None
    get_configured_select_async: (
        Callable[[Any, list[str] | None], Awaitable[tuple[Any, Any]]] | None
    ) = None
    async_sql_resource: Any | None = None
    chunk_size: int | None = None


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class BackendStrategy(ABC):
    """Interface that every backend must implement.

    Subclasses are **stateless** — a single singleton instance is shared
    across all :class:`~boti_data.gateway.core.DataGateway` instances.
    """

    @property
    @abstractmethod
    def name(self) -> BackendName:
        """Machine-readable backend identifier (e.g. ``"sqlalchemy"``)."""

    # -- Config construction -------------------------------------------------

    @abstractmethod
    def build_config(self, **kwargs: Any) -> BackendConfig:
        """Build a config object from keyword arguments (``from_backend``)."""

    @abstractmethod
    def build_config_from_dict(self, cfg: dict[str, Any]) -> BackendConfig:
        """Build a config object from a legacy-style dict (``from_config``).

        *cfg* has already had common gateway options (table, field_map, fs, …)
        removed by :func:`~._factory.extract_config_common_options`.
        """

    # -- Resource construction -----------------------------------------------

    @abstractmethod
    def build_resource(
        self,
        config: BackendConfig,
        *,
        fs: fsspec.AbstractFileSystem | None = None,
        fs_factory: Any | None = None,
    ) -> tuple[BackendName, BackendResource | None]:
        """Build the runtime resource for *config*.

        Returns ``(backend_name, resource_or_None)``.
        """

    # -- Config validation ---------------------------------------------------

    def validate_requirements(self, config: BackendConfig, **flags: Any) -> None:
        """Backend-specific config validation (no-op by default)."""

    # -- Structured-mode loads ------------------------------------------------

    @abstractmethod
    def load_structured_sync(self, ctx: StructuredLoadContext) -> FrameResult:
        """Execute a synchronous structured-mode load."""

    @abstractmethod
    async def load_structured_async(self, ctx: StructuredLoadContext) -> FrameResult:
        """Execute an asynchronous structured-mode load."""

    # -- Configured-mode loads -----------------------------------------------

    @abstractmethod
    def load_configured_sync(self, ctx: ConfiguredLoadContext) -> FrameResult:
        """Execute a synchronous configured-mode load."""

    @abstractmethod
    async def load_configured_async(self, ctx: ConfiguredLoadContext) -> FrameResult:
        """Execute an asynchronous configured-mode load."""

    # -- Async lifecycle hooks -----------------------------------------------

    async def setup_async_context(self, gateway: DataGateway) -> None:
        """Async context setup (e.g. opening an async SQL connection)."""

    async def teardown_async_context(self, gateway: DataGateway) -> None:
        """Async context teardown."""

    # -- Options preparation -------------------------------------------------

    def prepare_structured_options(
        self,
        options: dict[str, Any],
        field_map: FieldMap,
        configured: bool,
    ) -> dict[str, Any]:
        """Modify loader options before dispatch (e.g. field-map projection).

        No-op by default — only SQL overrides this.
        """
        return options

    # -- Chunking metadata ---------------------------------------------------

    def supports_chunking(self) -> bool:
        """Whether the backend supports IN-chunk fan-out."""
        return True

    def supports_in_chunk_hinting(self) -> bool:
        """Whether the backend uses SQL-style chunk hints."""
        return False

    def chunk_hint(
        self,
        filters: dict[str, Any],
        policy: InChunkPolicy,
    ) -> dict[str, Any] | None:
        """Return a chunking hint or ``None``.

        Only SQL overrides this.
        """
        return None

    # -- Auto return type ----------------------------------------------------

    def estimate_result_size(
        self,
        ctx: StructuredLoadContext | ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        """Return ``(estimated_rows, estimated_bytes)`` or ``None``.

        Used by :class:`~.return_type.AutoReturnTypeResolver` to decide
        whether a result should be eager (pandas) or lazy (dask).
        Returning ``None`` means "unknown — use the default."
        """
        return None

    async def estimate_result_size_async(
        self,
        ctx: StructuredLoadContext | ConfiguredLoadContext,
    ) -> tuple[int | None, int | None] | None:
        """Async variant of :meth:`estimate_result_size`.

        Default implementation delegates to the sync method.
        """
        return self.estimate_result_size(ctx)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STRATEGIES_BY_NAME: dict[str, BackendStrategy] = {}
_STRATEGIES_BY_CONFIG_TYPE: dict[type, BackendStrategy] = {}


def register(
    name: str,
    strategy: BackendStrategy,
    *config_types: type,
) -> None:
    """Register a strategy under *name* for the given *config_types*."""
    _STRATEGIES_BY_NAME[name] = strategy
    for ct in config_types:
        _STRATEGIES_BY_CONFIG_TYPE[ct] = strategy


def get(name: str) -> BackendStrategy:
    """Look up a strategy by name."""
    try:
        return _STRATEGIES_BY_NAME[name]
    except KeyError:
        raise ValueError(f"Unknown backend: {name!r}") from None


def for_config(config: BackendConfig) -> BackendStrategy:
    """Look up a strategy by config type."""
    strategy = _STRATEGIES_BY_CONFIG_TYPE.get(type(config))
    if strategy is None:
        raise ValueError(f"No strategy registered for {type(config).__name__}")
    return strategy


# ---------------------------------------------------------------------------
# Auto-register built-in strategies
# ---------------------------------------------------------------------------

# Imported down here (not at module top) since these sibling modules import
# BackendStrategy/StructuredLoadContext/ConfiguredLoadContext back from this
# module — importing them before those names exist would be circular.
from .datacube_strategy import DatacubeStrategy  # noqa: E402
from .parquet_strategy import ParquetStrategy  # noqa: E402
from .sql_strategy import SqlAlchemyStrategy  # noqa: E402

_SQLALCHEMY = SqlAlchemyStrategy()
_PARQUET = ParquetStrategy()
_DATACUBE = DatacubeStrategy()

register("sqlalchemy", _SQLALCHEMY, SqlDatabaseConfig)
register("parquet", _PARQUET, ParquetDataConfig)
register("datacube", _DATACUBE, DatacubeConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Unused in this module's own code; kept as a monkeypatch anchor
# (tests patch boti_data.gateway._backend_strategies._AUTO_EAGER_MAX_BYTES).
_AUTO_EAGER_MAX_BYTES = 32 * 1024 * 1024
