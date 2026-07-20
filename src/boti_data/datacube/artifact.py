"""Base ETL artifact with date window and before/after load hooks.

This module provides :class:`BaseArtifact`, an inheritance-based
extension point for Parquet-oriented data artifacts that need date
range partitioning and lifecycle hooks around the load step.

For the simpler datacube pattern without date windows, see
:class:`BaseDataCube`.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from boti.core import Logger
from boti.core.lifecycle import LifecycleCore
from boti.core.lifecycle_pickle import PicklableLifecycleCoreMixin

from boti_data.datacube.contract import DatacubeFrame, frame_has_data

if TYPE_CHECKING:
    from boti_data.helper import DataHelper

DateLike = str | date | datetime | None


def _validate_and_format_date(name: str, value: DateLike) -> str | None:
    """Normalize date-like input into a canonical ``'%Y-%m-%d'`` string.

    - ``None`` → ``None``
    - ``str`` / ``date`` / ``datetime`` → parsed via ``pd.to_datetime``,
      formatted as ``'%Y-%m-%d'``.
    - Anything else → ``TypeError``.
    """
    if value is None:
        return None
    if isinstance(value, (str, date, datetime)):
        try:
            return pd.to_datetime(value).date().strftime("%Y-%m-%d")
        except Exception as exc:
            raise ValueError(f"{name} must be a valid date, got {value!r}") from exc
    raise TypeError(f"{name} must be str, date, datetime, or None; got {type(value)}")


class BaseArtifact(PicklableLifecycleCoreMixin, LifecycleCore):
    """Base artifact for Parquet data with date window and load lifecycle hooks.

    Subclasses *may* override any of the four lifecycle hooks:

    - ``before_load(**kwargs)`` — sync, runs before loading.
    - ``after_load(**kwargs)`` — sync, runs after loading.
    - ``abefore_load(**kwargs)`` — async, runs before loading.
    - ``aafter_load(**kwargs)`` — async, runs after loading.

    Dates are always stored as canonical ``'%Y-%m-%d'`` strings.

    Usage::

        class OrdersArtifact(BaseArtifact):
            config = {
                "parquet_storage_path": "s3://bucket/orders",
                "return_type": "pandas",
            }

            def after_load(self, **kwargs):
                self.df = self.df[self.df["status"] == "shipped"]

        artifact = OrdersArtifact(
            parquet_start_date="2024-01-01",
            parquet_end_date="2024-06-30",
        )
        df = artifact.load()
    """

    df: DatacubeFrame | None = None
    config: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        from boti_data.helper import DataHelper

        merged = {**self.config, **kwargs}
        self._helper: DataHelper = DataHelper(merged)
        self.logger = kwargs.get("logger") or Logger.default_logger(
            logger_name=self.__class__.__name__
        )

        self.parquet_start_date: str | None = _validate_and_format_date(
            "parquet_start_date",
            merged.get("parquet_start_date"),
        )
        self.parquet_end_date: str | None = _validate_and_format_date(
            "parquet_end_date",
            merged.get("parquet_end_date"),
        )
        self.data_wrapper_class: type[Any] | None = merged.get("data_wrapper_class")
        self.class_params: dict[str, Any] = merged.get("class_params") or {
            "logger": self.logger,
        }
        self._validate_date_order()
        super().__init__()

    @classmethod
    def from_helper(cls, helper: DataHelper, **overrides: Any) -> BaseArtifact:
        """Construct from a pre-built :class:`DataHelper` instance."""
        instance = cls.__new__(cls)
        instance.df = None
        instance._helper = helper
        instance.config = {**cls.config, **overrides}
        instance.logger = overrides.get("logger") or Logger.default_logger(logger_name=cls.__name__)

        merged = {**cls.config, **overrides}
        instance.parquet_start_date = _validate_and_format_date(
            "parquet_start_date",
            merged.get("parquet_start_date"),
        )
        instance.parquet_end_date = _validate_and_format_date(
            "parquet_end_date",
            merged.get("parquet_end_date"),
        )
        instance.data_wrapper_class = merged.get("data_wrapper_class")
        instance.class_params = merged.get("class_params") or {
            "logger": instance.logger,
        }
        instance._validate_date_order()
        # __new__() bypasses __init__(), so LifecycleCore's state (_state_lock,
        # _is_closed, the GC finalizer, etc.) must be wired up explicitly here.
        LifecycleCore.__init__(instance)
        return instance

    def _validate_date_order(self) -> None:
        """Raise ``ValueError`` if start date is after end date."""
        if self.parquet_start_date and self.parquet_end_date:
            if self.parquet_start_date > self.parquet_end_date:
                raise ValueError(
                    f"parquet_start_date {self.parquet_start_date} "
                    f"cannot be after parquet_end_date {self.parquet_end_date}"
                )

    # ── optional hooks ──────────────────────────────────────────────────

    def before_load(self, **kwargs: Any) -> None:
        """Optional sync pre-load hook. Override in subclasses if needed."""
        return None

    def after_load(self, **kwargs: Any) -> None:
        """Optional sync post-load hook. Override in subclasses if needed."""
        return None

    async def abefore_load(self, **kwargs: Any) -> None:
        """Optional async pre-load hook. Override in subclasses if needed."""
        return None

    async def aafter_load(self, **kwargs: Any) -> None:
        """Optional async post-load hook. Override in subclasses if needed."""
        return None

    # ── internals ───────────────────────────────────────────────────────

    def _has_data(self) -> bool:
        return frame_has_data(self.df)

    # ── public API ──────────────────────────────────────────────────────

    # Not a copy-pasted twin: load()/aload() mirror each other only because
    # they call this class's separately-overridable sync/async hooks
    # (before_load/after_load vs abefore_load/aafter_load), which must stay
    # distinct so subclasses can override each independently.
    # spaghetti-ignore[sync-async-duplication]
    def load(self, **options: Any) -> DatacubeFrame:
        """Sync load path with ``before_load`` / ``after_load`` hooks."""
        self.before_load(**options)
        self.df = self._helper.load(**options)
        self.after_load(**options)
        return self.df

    async def aload(self, **options: Any) -> DatacubeFrame:
        """Async load path with ``abefore_load`` / ``aafter_load`` hooks."""
        await self.abefore_load(**options)
        self.df = await self._helper.aload(**options)
        await self.aafter_load(**options)
        return self.df

    # ── date window ─────────────────────────────────────────────────────

    def has_date_window(self) -> bool:
        """Return ``True`` if either start or end date is set."""
        return bool(self.parquet_start_date or self.parquet_end_date)

    def date_window(self) -> tuple[str | None, str | None]:
        """Return ``(start_date, end_date)`` as canonical strings."""
        return self.parquet_start_date, self.parquet_end_date

    def to_params(self) -> dict[str, Any]:
        """Return a serializable dict of artifact parameters."""
        return {
            "parquet_start_date": self.parquet_start_date,
            "parquet_end_date": self.parquet_end_date,
            "data_wrapper_class": self.data_wrapper_class,
            "class_params": dict(self.class_params),
        }

    # ── lifecycle ───────────────────────────────────────────────────────

    def __enter__(self) -> BaseArtifact:
        super().__enter__()
        self._helper.__enter__()
        return self

    async def __aenter__(self) -> BaseArtifact:
        await super().__aenter__()
        await self._helper.__aenter__()
        return self

    def _cleanup(self) -> None:
        self._helper.close()

    async def _acleanup(self) -> None:
        await self._helper.aclose()


__all__ = ["BaseArtifact", "DateLike", "_validate_and_format_date"]
