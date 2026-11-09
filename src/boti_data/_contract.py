"""Public API contract helpers: stability labels and deprecations.

``boti_data.__all__`` is split into :data:`boti_data.STABLE` and
:data:`boti_data.EXPERIMENTAL`. Stable names follow semantic versioning: they
change shape only in a major release, and are first deprecated for at least
one minor release. Experimental names may change in any minor release and are
documented as such. See CHANGELOG.md, "Versioning policy".
"""

from __future__ import annotations

import warnings

__all__ = ["Deprecation", "deprecated"]


class Deprecation:
    """Where a public name is going: the release that removes it and, if any,
    the name to use instead."""

    __slots__ = ("removal", "replacement")

    def __init__(self, removal: str, replacement: str | None = None) -> None:
        self.removal = removal
        self.replacement = replacement

    def message(self, name: str) -> str:
        text = f"boti_data.{name} is deprecated and will be removed in boti-data {self.removal}"
        if self.replacement:
            text += f"; use {self.replacement} instead"
        return text + "."


def deprecated(name: str, *, removal: str, replacement: str | None = None) -> None:
    """Warn that *name* is deprecated and goes away in release *removal*.

    Called by the package's lazy ``__getattr__`` for every name listed in
    ``boti_data.DEPRECATED``, so the warning fires on first access, where the
    caller's own frame is reported (``stacklevel=3``: this helper, the module
    ``__getattr__``, then the importer).
    """
    warnings.warn(
        Deprecation(removal, replacement).message(name),
        DeprecationWarning,
        stacklevel=3,
    )
