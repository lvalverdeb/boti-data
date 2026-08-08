"""
Regression tests: a DataGateway whose __init__ rejects the configuration must
not leave the backend resource it already built open.

``build_gateway_state()`` constructs the resource before running the validations
that can reject a configuration. When one of those rejects, the gateway's
``LifecycleCore.__init__`` never runs, so nothing owns the resource and nothing
will ever close it — it survives until the GC finalizer logs a "was garbage
collected without being closed" warning. That warning is accurate: the resource
genuinely leaked, and in a real process it holds its handles until collection.
"""

from __future__ import annotations

import pandas as pd
import pytest

from boti_data.datacube.contract import DatacubeConfig
from boti_data.datacube.resource import DatacubeResource
from boti_data.gateway import DataGateway


def _loader(_request) -> pd.DataFrame:
    return pd.DataFrame({"ok": [True]})


def test_resource_is_closed_when_init_rejects_the_configuration() -> None:
    # Capture the resource the gateway builds, so it can be inspected after the
    # constructor has raised and dropped its own reference to it.
    captured: list[DatacubeResource] = []
    original_close = DatacubeResource.close

    def _tracking_close(self, **kwargs):
        captured.append(self)
        return original_close(self, **kwargs)

    DatacubeResource.close = _tracking_close  # type: ignore[method-assign]
    try:
        with pytest.raises(ValueError, match="require_datacube_request_validator=True"):
            DataGateway(
                DatacubeConfig(loader=_loader, default_cube="sales"),
                require_datacube_request_validator=True,
            )
    finally:
        DatacubeResource.close = original_close  # type: ignore[method-assign]

    assert captured, "the resource built before validation failed was never closed"
    assert all(resource.closed for resource in captured), (
        "close() was called but the resource did not end up closed"
    )


def _run_with_exploding_close() -> tuple[list[DatacubeResource], list[str]]:
    """Build a gateway that fails validation *and* whose close() also fails."""
    original_close = DatacubeResource.close
    orphaned: list[DatacubeResource] = []
    warnings: list[str] = []

    def _exploding_close(self, **kwargs):
        orphaned.append(self)
        logger = getattr(self, "logger", None)
        if logger is not None:
            original_warning = logger.warning
            logger.warning = lambda msg, *a, **kw: warnings.append(str(msg))
            self._restore_warning = (logger, original_warning)
        raise RuntimeError("close blew up")

    DatacubeResource.close = _exploding_close  # type: ignore[method-assign]
    try:
        # The ValueError from validation must survive, not the RuntimeError.
        with pytest.raises(ValueError, match="require_datacube_request_validator=True"):
            DataGateway(
                DatacubeConfig(loader=_loader, default_cube="sales"),
                require_datacube_request_validator=True,
            )
    finally:
        DatacubeResource.close = original_close  # type: ignore[method-assign]
        for resource in orphaned:
            restore = getattr(resource, "_restore_warning", None)
            if restore is not None:
                logger, original_warning = restore
                logger.warning = original_warning
            # This test deliberately sabotages close(), so the resource really
            # is left open. Close it properly now that the real method is back,
            # or its GC finalizer warns and re-adds the noise this suite removed.
            original_close(resource)
    return orphaned, warnings


def test_original_configuration_error_is_not_masked_by_cleanup() -> None:
    """Cleanup runs while an exception propagates; it must not replace it."""
    orphaned, _ = _run_with_exploding_close()
    assert orphaned, "cleanup never attempted to close the orphaned resource"


def test_a_failing_cleanup_close_is_reported_not_swallowed() -> None:
    """A close() that also fails means a handle really leaked — say so."""
    _, warnings = _run_with_exploding_close()

    assert warnings, "the failed close() was silently discarded"
    assert any("may have leaked" in message for message in warnings), warnings
    assert any("close blew up" in message for message in warnings), warnings
