from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from boti_data.schema import DataFrameLike
from boti_data.watermark.incremental import (
    IncrementalResult,
    advance_watermark,
    build_incremental_filters,
)

if TYPE_CHECKING:
    from boti_data.gateway import DataGateway


@runtime_checkable
class WatermarkStore(Protocol):
    """Duck-typed store with ``read`` / ``write`` for bookmark persistence."""

    def read(self, *, source: str) -> Any: ...

    def write(self, *, source: str, value: Any) -> None: ...


def _prepare_watermark(
    watermark_store: WatermarkStore,
    watermark_source: str,
    initial_value: Any,
    watermark_field: str,
    operator: str,
    raw_options: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    opts = dict(raw_options)
    watermark_value = watermark_store.read(source=watermark_source) or initial_value
    if watermark_value is not None:
        filters = build_incremental_filters(
            watermark_field=watermark_field,
            watermark_value=watermark_value,
            operator=operator,
        )
        merged = {**filters, **opts.pop("filters", {})}
        opts["filters"] = merged
    return watermark_value, opts


def _finalize_incremental(
    frame: Any,
    *,
    watermark_field: str,
    previous_watermark: Any,
    watermark_source: str,
    watermark_store: WatermarkStore,
    commit_on_success: bool,
) -> IncrementalResult:
    if hasattr(frame, "compute"):
        frame = frame.compute()
    current = advance_watermark(frame, watermark_field=watermark_field)
    committed = False
    if current is not None and previous_watermark != current and commit_on_success:
        watermark_store.write(source=watermark_source, value=current)
        committed = True
    frame_like: DataFrameLike = frame
    if hasattr(frame_like, "to_pandas"):
        frame_like = frame_like.to_pandas()
    return IncrementalResult(
        frame=frame_like,
        watermark_field=watermark_field,
        previous_watermark=previous_watermark,
        current_watermark=current,
        records_loaded=len(frame_like) if hasattr(frame_like, "__len__") else 0,
        watermark_committed=committed,
    )


class IncrementalLoadService:
    """Encapsulates incremental-data-load orchestration with watermark tracking.

    Separates watermark bookkeeping (read/write) from data loading, keeping the
    facade layer (``DataHelper``) free of business logic.
    """

    def __init__(self, gateway: DataGateway) -> None:
        self._gateway = gateway

    # Not a copy-pasted twin: both already share _prepare_watermark()/
    # _finalize_incremental(); the only remaining difference is
    # self._gateway.load(**opts) vs await self._gateway.aload(**opts) —
    # the theoretical floor for this pattern.
    # spaghetti-ignore[sync-async-duplication]: see above
    def load(
        self,
        *,
        watermark_field: str,
        watermark_source: str,
        watermark_store: WatermarkStore,
        operator: str = "gt",
        initial_value: Any = None,
        commit_on_success: bool = True,
        **options: Any,
    ) -> IncrementalResult:
        watermark_value, opts = _prepare_watermark(
            watermark_store,
            watermark_source,
            initial_value,
            watermark_field,
            operator,
            options,
        )
        frame = self._gateway.load(**opts)
        return _finalize_incremental(
            frame,
            watermark_field=watermark_field,
            previous_watermark=watermark_value,
            watermark_source=watermark_source,
            watermark_store=watermark_store,
            commit_on_success=commit_on_success,
        )

    async def aload(
        self,
        *,
        watermark_field: str,
        watermark_source: str,
        watermark_store: WatermarkStore,
        operator: str = "gt",
        initial_value: Any = None,
        commit_on_success: bool = True,
        **options: Any,
    ) -> IncrementalResult:
        watermark_value, opts = _prepare_watermark(
            watermark_store,
            watermark_source,
            initial_value,
            watermark_field,
            operator,
            options,
        )
        frame = await self._gateway.aload(**opts)
        return _finalize_incremental(
            frame,
            watermark_field=watermark_field,
            previous_watermark=watermark_value,
            watermark_source=watermark_source,
            watermark_store=watermark_store,
            commit_on_success=commit_on_success,
        )


_SINCE_TUPLE_ARITY = 3


@dataclass(frozen=True)
class SinceSpec:
    """What ``DataGateway.load(since=...)`` needs: the field, the store and the source key."""

    field: str
    store: WatermarkStore
    source: str
    operator: str = "gt"
    initial_value: Any = None


def resolve_since(since: Any, options: dict[str, Any], *, default_source: str | None) -> SinceSpec:
    """Build a :class:`SinceSpec` from ``since`` plus the ``watermark_*`` keywords in ``options``.

    ``since`` is either a watermark field name (then ``watermark_store=`` is
    required and ``watermark_source=`` defaults to the configured table) or a
    ``(field, store, source)`` tuple. The keywords are removed from ``options``.
    """
    store = options.pop("watermark_store", None)
    source = options.pop("watermark_source", None)
    operator = options.pop("watermark_operator", "gt")
    initial_value = options.pop("watermark_initial", None)
    if isinstance(since, tuple):
        if len(since) != _SINCE_TUPLE_ARITY:
            raise ValueError("since must be a field name or a (field, store, source) tuple.")
        field, store, source = since
    elif isinstance(since, str):
        field = since
    else:
        raise TypeError("since must be a field name or a (field, store, source) tuple.")
    source = source or default_source
    if store is None or not isinstance(store, WatermarkStore):
        raise ValueError(
            "since= needs a watermark store with read()/write(); pass watermark_store=."
        )
    if not source:
        raise ValueError(
            "since= needs a watermark source; pass watermark_source= or configure a table."
        )
    return SinceSpec(
        field=field, store=store, source=source, operator=operator, initial_value=initial_value
    )


def _since_call(options: dict[str, Any], *, default_source: str | None) -> dict[str, Any]:
    """Keyword arguments for :class:`IncrementalLoadService` from a ``since=`` option set."""
    spec = resolve_since(options.pop("since"), options, default_source=default_source)
    return {
        "watermark_field": spec.field,
        "watermark_source": spec.source,
        "watermark_store": spec.store,
        "operator": spec.operator,
        "initial_value": spec.initial_value,
        **options,
    }


# Not a copy-pasted twin: everything but the await lives in _since_call().
# spaghetti-ignore[sync-async-duplication]: see above
def load_since(gateway: DataGateway, options: dict[str, Any], *, default_source: str | None) -> Any:
    """``DataGateway.load(since=...)``: load above the watermark and advance it on success."""
    call = _since_call(options, default_source=default_source)
    return IncrementalLoadService(gateway).load(**call).frame


async def aload_since(
    gateway: DataGateway, options: dict[str, Any], *, default_source: str | None
) -> Any:
    """Async twin of :func:`load_since`."""
    call = _since_call(options, default_source=default_source)
    result = await IncrementalLoadService(gateway).aload(**call)
    return result.frame
