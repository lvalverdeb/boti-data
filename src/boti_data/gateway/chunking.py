from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa

from boti_data.gateway._backend_strategies import BackendStrategy

from .frame_strategies import FrameResult, get_frame_strategy
from .normalization import DEFAULT_IN_CHUNK_SIZE, LOAD_CONTROL_KEYS, split_control_and_filters
from .planning import InChunkPolicy
from .requests import ReturnType

ChunkTarget = tuple[str, str, Any]
ExecuteFn = Callable[..., FrameResult]
AsyncExecuteFn = Callable[..., Any]
PolicyProvider = Callable[[], InChunkPolicy]


class InChunkPlanner:
    """Determines when gateway loads should fan out large ``IN`` filters."""

    def __init__(
        self,
        *,
        strategy: BackendStrategy,
        policy_provider: PolicyProvider,
    ) -> None:
        self._strategy = strategy
        self._policy_provider = policy_provider

    def resolve_controls(
        self,
        options: dict[str, Any],
        *,
        strategy: str,
        execution_mode: str | None,
        in_chunk_size_raw: Any,
        in_chunk_concurrency_raw: Any,
    ) -> tuple[int, int | None]:
        if in_chunk_size_raw is not None or in_chunk_concurrency_raw is not None:
            return self._explicit_controls(in_chunk_size_raw, in_chunk_concurrency_raw)
        if strategy == "off":
            return 0, None
        if strategy != "auto":
            raise ValueError("in_chunk_strategy must be 'auto' or 'off'.")

        filters = self.extract_filter_mapping(options)
        return self._auto_controls(filters, execution_mode=execution_mode)

    @staticmethod
    def _explicit_controls(
        in_chunk_size_raw: Any,
        in_chunk_concurrency_raw: Any,
    ) -> tuple[int, int | None]:
        size = DEFAULT_IN_CHUNK_SIZE if in_chunk_size_raw is None else int(in_chunk_size_raw)
        concurrency = None if in_chunk_concurrency_raw is None else int(in_chunk_concurrency_raw)
        return size, concurrency

    def _auto_controls(
        self,
        filters: dict[str, Any],
        *,
        execution_mode: str | None,
    ) -> tuple[int, int | None]:
        if not self._strategy.supports_chunking():
            return 0, None

        hint = None
        if self._strategy.supports_in_chunk_hinting() and filters:
            policy = self._policy_provider()
            hint = self._strategy.chunk_hint(filters, policy)
        if execution_mode == "eager":
            return self._eager_controls(filters, hint)
        if hint is not None:
            return int(hint["in_chunk_size"]), int(hint["in_chunk_concurrency"])
        return DEFAULT_IN_CHUNK_SIZE, None

    @staticmethod
    def _eager_controls(
        filters: dict[str, Any],
        hint: dict[str, Any] | None,
    ) -> tuple[int, int | None]:
        if InChunkPlanner.max_in_filter_size(filters) < DEFAULT_IN_CHUNK_SIZE and hint is not None:
            return int(hint["in_chunk_size"]), int(hint["in_chunk_concurrency"])
        return 0, None

    @staticmethod
    def max_in_filter_size(filters: dict[str, Any]) -> int:
        sizes = [
            len(value)
            for key, value in filters.items()
            if key.endswith("__in") and isinstance(value, (list, tuple))
        ]
        return max(sizes, default=0)

    @staticmethod
    def extract_filter_mapping(options: dict[str, Any]) -> dict[str, Any]:
        nested_filters = options.get("filters")
        if isinstance(nested_filters, dict):
            return nested_filters
        _, runtime_filters = split_control_and_filters(options)
        return runtime_filters


class ChunkedLoadExecutor:
    """Executes a load once or as multiple chunked ``IN`` loads."""

    @classmethod
    async def aload(
        cls,
        execute_fn: AsyncExecuteFn,
        chunk_size: int,
        options: dict[str, Any],
        *,
        return_type: ReturnType,
        max_concurrency: int | None = None,
    ) -> FrameResult:
        if chunk_size <= 0:
            return await execute_fn(**options)

        chunk_target = cls.find_chunk_target(options, chunk_size=chunk_size)
        if chunk_target is None:
            return await execute_fn(**options)

        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency and max_concurrency > 0 else None
        tasks = [
            cls._run_async_chunk(execute_fn, chunk_opts, semaphore)
            for chunk_opts in cls.iter_chunk_option_sets(options, chunk_target, chunk_size)
        ]
        return cls._combine_results(
            await asyncio.gather(*tasks),
            return_type=return_type,
            persist_dask=True,
        )

    @classmethod
    def load(
        cls,
        execute_fn: ExecuteFn,
        chunk_size: int,
        options: dict[str, Any],
        *,
        return_type: ReturnType,
        max_concurrency: int | None = None,
        logger: Any | None = None,
    ) -> FrameResult:
        if chunk_size <= 0:
            return execute_fn(**options)

        chunk_target = cls.find_chunk_target(options, chunk_size=chunk_size)
        if chunk_target is None:
            return execute_fn(**options)

        chunk_opts_list = list(cls.iter_chunk_option_sets(options, chunk_target, chunk_size))
        cls._log_sync_fanout(
            options,
            logger=logger,
            chunk_count=len(chunk_opts_list),
            chunk_target=chunk_target,
            chunk_size=chunk_size,
            max_concurrency=max_concurrency,
        )
        results = cls._execute_sync_chunks(execute_fn, chunk_opts_list, max_concurrency)
        return cls._combine_results(results, return_type=return_type, persist_dask=True)

    @staticmethod
    async def _run_async_chunk(
        execute_fn: AsyncExecuteFn,
        chunk_opts: dict[str, Any],
        semaphore: asyncio.Semaphore | None,
    ) -> FrameResult:
        if semaphore is None:
            return await execute_fn(**chunk_opts)
        async with semaphore:
            return await execute_fn(**chunk_opts)

    @staticmethod
    def _execute_sync_chunks(
        execute_fn: ExecuteFn,
        chunk_opts_list: list[dict[str, Any]],
        max_concurrency: int | None,
    ) -> list[FrameResult]:
        if max_concurrency is not None and max_concurrency > 1:
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                futures = [executor.submit(execute_fn, **chunk_opts) for chunk_opts in chunk_opts_list]
                return [future.result() for future in futures]
        return [execute_fn(**chunk_opts) for chunk_opts in chunk_opts_list]

    @classmethod
    def _combine_results(
        cls,
        results: list[FrameResult],
        *,
        return_type: ReturnType,
        persist_dask: bool,
    ) -> FrameResult:
        non_empty = [result for result in results if cls.has_any_rows(result)]
        if not non_empty:
            return results[0]
        if len(non_empty) == 1:
            return non_empty[0]

        combined = get_frame_strategy(return_type).concat(non_empty)
        if persist_dask and isinstance(combined, dd.DataFrame):
            return get_frame_strategy("dask").persist(combined)
        return combined

    @staticmethod
    def _log_sync_fanout(
        options: dict[str, Any],
        *,
        logger: Any | None,
        chunk_count: int,
        chunk_target: ChunkTarget,
        chunk_size: int,
        max_concurrency: int | None,
    ) -> None:
        if not options.get("diagnostics") or logger is None:
            return
        logger.info(
            "Gateway IN chunk fan-out sync: %d chunks, target=%s, "
            "chunk_size=%d, concurrency=%s",
            chunk_count,
            chunk_target,
            chunk_size,
            max_concurrency,
        )

    @staticmethod
    def is_chunkable_in_value(value: Any, *, chunk_size: int) -> bool:
        return (
            hasattr(value, "__len__")
            and hasattr(value, "__getitem__")
            and not isinstance(value, (str, bytes, pd.Series, dd.Series, pl.Series))
            and len(value) > chunk_size
        )

    @classmethod
    def find_chunk_target(
        cls,
        options: dict[str, Any],
        *,
        chunk_size: int,
    ) -> ChunkTarget | None:
        for key, value in options.items():
            if key in LOAD_CONTROL_KEYS:
                continue
            if key.endswith("__in") and cls.is_chunkable_in_value(value, chunk_size=chunk_size):
                return ("top_level", key, value)

        nested_filters = options.get("filters")
        if isinstance(nested_filters, dict):
            for key, value in nested_filters.items():
                if key.endswith("__in") and cls.is_chunkable_in_value(value, chunk_size=chunk_size):
                    return ("filters", key, value)
        return None

    @staticmethod
    def iter_chunk_option_sets(
        options: dict[str, Any],
        chunk_target: ChunkTarget,
        chunk_size: int,
    ) -> Iterable[dict[str, Any]]:
        scope, key, values = chunk_target
        for index in range(0, len(values), chunk_size):
            chunk_opts = dict(options)
            if scope == "top_level":
                chunk_opts[key] = values[index : index + chunk_size]
            else:
                nested_filters = dict(chunk_opts.get("filters", {}))
                nested_filters[key] = values[index : index + chunk_size]
                chunk_opts["filters"] = nested_filters
            yield chunk_opts

    @staticmethod
    def has_any_rows(frame: FrameResult) -> bool:
        try:
            return _strategy_for_frame(frame).has_any_rows(frame)
        except Exception:
            return False


def _strategy_for_frame(frame: FrameResult):
    if isinstance(frame, dd.DataFrame):
        return get_frame_strategy("dask")
    if isinstance(frame, pd.DataFrame):
        return get_frame_strategy("pandas")
    if isinstance(frame, pa.Table):
        return get_frame_strategy("arrow")
    if isinstance(frame, pl.DataFrame):
        return get_frame_strategy("polars")
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")
