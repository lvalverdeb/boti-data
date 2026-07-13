"""Base datacube with mutable .df and subclass-overridable transform hooks.

This module provides :class:`BaseDataCube`, an inheritance-based extension
point for building data containers that load data via :class:`DataHelper`
and optionally apply sync/async transforms after loading.

For composition-based datacube patterns, see :class:`DatacubeResource` and
:class:`DatacubeContract`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from boti.core import Logger
from boti.core.lifecycle import LifecycleCore
from boti.core.lifecycle_pickle import PicklableLifecycleCoreMixin

from boti_data.datacube.contract import DatacubeFrame

if TYPE_CHECKING:
    from boti_data.helper import DataHelper


class BaseDataCube(PicklableLifecycleCoreMixin, LifecycleCore):
    """Base datacube with mutable ``.df`` and subclass-overridable transform hooks.

    Subclasses *may* override:

    - ``fix_data(**kwargs)`` — synchronous, local transforms.
    - ``afix_data(**kwargs)`` — asynchronous transforms (I/O, awaits).

    Semantics:

    - ``load()`` → delegates to the internal :class:`DataHelper`, then runs
      ``fix_data()`` if the subclass overrides it.
    - ``aload()`` → delegates to the internal :class:`DataHelper`, then runs
      ``afix_data()`` if overridden, else falls back to ``fix_data()``.

    Usage::

        class MyCube(BaseDataCube):
            config = {"table": "orders", "return_type": "pandas"}

            def fix_data(self, **kwargs):
                self.df = self.df[self.df["status"] == "active"]

        cube = MyCube(dsn="postgresql://...")
        df = cube.load()  # loads + filters
    """

    df: DatacubeFrame | None = None
    config: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        from boti_data.helper import DataHelper

        merged = {**self.config, **kwargs}
        self._helper: DataHelper = DataHelper(merged)
        self.logger = Logger.default_logger(logger_name=self.__class__.__name__)
        super().__init__()

    @classmethod
    def from_helper(cls, helper: DataHelper, **overrides: Any) -> BaseDataCube:
        """Construct from a pre-built :class:`DataHelper` instance."""
        instance = cls.__new__(cls)
        instance.df = None
        instance._helper = helper
        instance.config = {**cls.config, **overrides}
        instance.logger = Logger.default_logger(logger_name=cls.__name__)
        # __new__() bypasses __init__(), so LifecycleCore's state (_state_lock,
        # _is_closed, the GC finalizer, etc.) must be wired up explicitly here.
        LifecycleCore.__init__(instance)
        return instance

    # ── optional hooks ──────────────────────────────────────────────────

    def fix_data(self, **kwargs: Any) -> None:
        """Optional sync transform hook. Override in subclasses if needed."""
        return None

    async def afix_data(self, **kwargs: Any) -> None:
        """Optional async transform hook. Override in subclasses if needed."""
        return None

    # ── internals ───────────────────────────────────────────────────────

    def _has_data(self) -> bool:
        """Check if dataframe has rows; avoids hidden heavy ops where possible."""
        if self.df is None:
            return False
        if isinstance(self.df, dd.DataFrame):
            return len(self.df.index) > 0
        if isinstance(self.df, pd.DataFrame):
            return len(self.df.index) > 0
        if isinstance(self.df, pa.Table):
            return len(self.df) > 0
        if hasattr(self.df, "height"):
            return self.df.height > 0
        return True

    def _afix_data_is_overridden(self) -> bool:
        """Check if subclass provided its own ``afix_data``."""
        return self.__class__.afix_data is not BaseDataCube.afix_data

    def _fix_data_is_overridden(self) -> bool:
        """Check if subclass provided its own ``fix_data``."""
        return self.__class__.fix_data is not BaseDataCube.fix_data

    # ── public API ──────────────────────────────────────────────────────

    def load(self, **options: Any) -> DatacubeFrame:
        """Sync load path with optional ``fix_data`` hook."""
        self.df = self._helper.load(**options)
        if self._has_data() and self._fix_data_is_overridden():
            self.fix_data()
        elif not self._has_data():
            self.logger.debug(
                "No data was found by %s loader", self.__class__.__name__
            )
        return self.df

    async def aload(self, **options: Any) -> DatacubeFrame:
        """Async load path with optional ``afix_data`` / ``fix_data`` hook."""
        self.df = await self._helper.aload(**options)
        if self._has_data():
            if self._afix_data_is_overridden():
                await self.afix_data()
            elif self._fix_data_is_overridden():
                self.fix_data()
        else:
            self.logger.debug(
                "No data was found by %s loader", self.__class__.__name__
            )
        return self.df

    # ── lifecycle ───────────────────────────────────────────────────────

    def __enter__(self) -> BaseDataCube:
        super().__enter__()
        self._helper.__enter__()
        return self

    async def __aenter__(self) -> BaseDataCube:
        await super().__aenter__()
        await self._helper.__aenter__()
        return self

    def _cleanup(self) -> None:
        self._helper.close()

    async def _acleanup(self) -> None:
        await self._helper.aclose()


__all__ = ["BaseDataCube"]
