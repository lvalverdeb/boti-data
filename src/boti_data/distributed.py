from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
from boti.core.logger import Logger

try:
    from dask.distributed import Client, LocalCluster, get_client
except ImportError:  # pragma: no cover - dependency is installed in normal environments
    Client = None  # type: ignore[assignment]
    LocalCluster = None  # type: ignore[assignment]
    get_client = None  # type: ignore[assignment]

# Shared-session pool keyed by shared_key
_dask_session_pool: dict[Any, Any] = {}


@dataclass(slots=True)
class DaskSession:
    """Explicit Dask client/session helper with optional cluster creation."""

    client: Any | None = None
    scheduler_address: str | None = None
    cluster_factory: Callable[..., Any] | None = None
    cluster_kwargs: Mapping[str, Any] = field(default_factory=dict)
    client_kwargs: Mapping[str, Any] = field(default_factory=dict)
    verify_connectivity: bool = False
    shared: bool = False
    shared_key: str | None = None
    logger: Any | None = None

    _cluster: Any | None = field(init=False, default=None)
    _owns_client: bool = field(init=False, default=False)
    _owns_cluster: bool = field(init=False, default=False)

    def __enter__(self) -> Any:
        return self.open()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> Any:
        return await asyncio.to_thread(self.open)

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    def open(self) -> Any:
        if self.client is not None:
            self._log("debug", f"Using external Dask client {describe_client(self.client)}")
            return self.client

        if self.shared and self.shared_key is not None:
            if self.shared_key in _dask_session_pool:
                existing = _dask_session_pool[self.shared_key]
                self.client = existing.client
                self._log(
                    "debug",
                    f"Reusing shared Dask client for key '{self.shared_key}'",
                )
                return self.client

        if Client is None:
            raise RuntimeError(
                "dask.distributed is required for DaskSession. Install dask[distributed] first."
            )

        if self.scheduler_address is not None:
            self.client = Client(self.scheduler_address, **dict(self.client_kwargs))
            self._owns_client = True
            self._log("info", f"Connected Dask client to {describe_client(self.client)}")
            if self.shared and self.shared_key is not None:
                _dask_session_pool[self.shared_key] = self
            return self.client

        cluster_factory = self.cluster_factory or LocalCluster
        if cluster_factory is None:
            raise RuntimeError(
                "LocalCluster is unavailable. Install dask[distributed] to create a managed session."
            )

        self._cluster = cluster_factory(**dict(self.cluster_kwargs))
        self._owns_cluster = True
        self.client = Client(self._cluster, **dict(self.client_kwargs))
        self._owns_client = True
        self._log("info", f"Started managed Dask session {describe_client(self.client)}")
        if self.shared and self.shared_key is not None:
            _dask_session_pool[self.shared_key] = self
        return self.client

    def close(self) -> None:
        if self.shared and self.shared_key is not None:
            _dask_session_pool.pop(self.shared_key, None)
        try:
            if self._owns_client and self.client is not None:
                self.client.close()
        finally:
            self.client = None
            self._owns_client = False
            if self._owns_cluster and self._cluster is not None:
                self._cluster.close()
            self._cluster = None
            self._owns_cluster = False

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def _log(self, level: str, message: str) -> None:
        logger = self.logger
        if logger is None:
            return
        log_fn = getattr(logger, level, None)
        if callable(log_fn):
            log_fn(message)


def dask_session(**kwargs: Any) -> DaskSession:
    """Factory for :class:`DaskSession`."""

    return DaskSession(**kwargs)


def describe_client(client: Any) -> dict[str, Any]:
    """Return a compact client summary for logging/debug output."""

    try:
        info = client.scheduler_info()
    except Exception:
        info = {}
    workers = info.get("workers", {}) if isinstance(info, dict) else {}
    scheduler = getattr(getattr(client, "cluster", None), "scheduler_address", None)
    if scheduler is None:
        scheduler = getattr(getattr(client, "scheduler", None), "address", None)
    return {
        "scheduler": scheduler,
        "dashboard": getattr(client, "dashboard_link", None),
        "workers": len(workers),
        "threads": sum(worker.get("nthreads", 0) for worker in workers.values()),
    }


def current_client_summary() -> dict[str, Any] | None:
    if get_client is None:
        return None
    try:
        return describe_client(get_client())
    except Exception:
        return None


def describe_frame(frame: Any) -> dict[str, Any]:
    """Return a compact frame summary suitable for diagnostics logs."""

    if isinstance(frame, dd.DataFrame):
        graph = frame.__dask_graph__()
        return {
            "engine": "dask",
            "columns": len(frame.columns),
            "npartitions": frame.npartitions,
            "graph_tasks": len(graph) if hasattr(graph, "__len__") else None,
            "graph_layers": len(frame.dask.layers) if hasattr(frame.dask, "layers") else None,
            "known_divisions": frame.known_divisions,
        }
    if isinstance(frame, pd.DataFrame):
        return {
            "engine": "pandas",
            "rows": len(frame.index),
            "columns": len(frame.columns),
        }
    if isinstance(frame, pa.Table):
        return {
            "engine": "arrow",
            "rows": frame.num_rows,
            "columns": len(frame.column_names),
        }
    if isinstance(frame, pl.DataFrame):
        return {
            "engine": "polars",
            "rows": frame.height,
            "columns": frame.width,
        }
    return {"engine": type(frame).__name__}


def diagnostics_logger(logger: Any | None, *, name: str) -> Any:
    return logger or Logger.default_logger(logger_name=name)


__all__ = [
    "DaskSession",
    "current_client_summary",
    "dask_session",
    "describe_client",
    "describe_frame",
    "diagnostics_logger",
]
